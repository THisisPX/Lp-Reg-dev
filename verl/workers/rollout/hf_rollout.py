# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import os

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


def visualize_sequence_entropy(
    generated_ids: list[int],
    step_logits: torch.Tensor,
    tokenizer,
    beta: float,
    tau: float,
    sample_id: int,
) -> None:
    """Visualize token-level entropy along generated sequence."""
    if len(generated_ids) == 0 or step_logits.numel() == 0:
        return

    step_logits = step_logits.float().detach().cpu()
    probs = F.softmax(step_logits, dim=-1)

    token_indices = torch.tensor(generated_ids, dtype=torch.long)
    token_prob = probs.gather(1, token_indices.unsqueeze(1)).squeeze(1)

    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    token_texts = tokenizer.convert_ids_to_tokens(generated_ids)

    categories = []
    colors = []
    for p in token_prob.tolist():
        if p >= beta:
            categories.append("normal")
            colors.append("gray")
        elif p >= tau:
            categories.append("reserved")
            colors.append("red")
        else:
            categories.append("filtered")
            colors.append("yellow")

    output_dir = os.path.join("outputs", "sequence_entropy")
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"sample_{sample_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("token\tprob\tentropy\tcategory\n")
        for token, p, h, cat in zip(token_texts, token_prob.tolist(), entropy.tolist(), categories, strict=False):
            f.write(f"{token}\t{p:.6f}\t{h:.6f}\t{cat}\n")

    try:
        import matplotlib.pyplot as plt

        positions = list(range(len(generated_ids)))
        plt.figure(figsize=(max(8, len(generated_ids) * 0.3), 4))
        plt.scatter(positions, entropy.tolist(), c=colors, s=25)
        plt.plot(positions, entropy.tolist(), color="black", linewidth=0.8, alpha=0.6)
        plt.xlabel("sequence position")
        plt.ylabel("token entropy")
        plt.xticks(positions, token_texts, rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{sample_id}.png"), dpi=200)
        plt.close()
    except Exception:
        # Keep generation robust if matplotlib is unavailable.
        pass


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        max_samples = self.config.get("sequence_entropy_max_samples", 3)
        visualized = 0
        output = []
        for p in batch_prompts:
            remaining = max(0, max_samples - visualized)
            out, newly_visualized = self._generate_minibatch(p, sample_id_start=visualized, max_samples=remaining)
            output.append(out)
            visualized += newly_visualized
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto, sample_id_start: int = 0, max_samples: int = 3) -> tuple[DataProto, int]:
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm
        beta = prompts.meta_info.get("sequence_entropy_beta", self.config.get("sequence_entropy_beta", 0.5))
        tau = prompts.meta_info.get("sequence_entropy_tau", self.config.get("sequence_entropy_tau", 0.1))

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        visualized_count = 0
        if self.tokenizer is not None and max_samples > 0 and output.scores is not None:
            num_steps = len(output.scores)
            if num_steps > 0:
                step_logits = torch.stack(output.scores, dim=0)  # [sequence_length, batch, vocab]
                samples_to_process = min(max_samples, generated_batch_size)
                for sample_offset in range(samples_to_process):
                    generated_ids = response[sample_offset, :num_steps].detach().cpu().tolist()
                    visualize_sequence_entropy(
                        generated_ids=generated_ids,
                        step_logits=step_logits[:, sample_offset, :],
                        tokenizer=self.tokenizer,
                        beta=beta,
                        tau=tau,
                        sample_id=sample_id_start + sample_offset,
                    )
                    visualized_count += 1

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch), visualized_count

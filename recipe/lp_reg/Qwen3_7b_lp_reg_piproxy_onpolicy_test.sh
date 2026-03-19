#!/usr/bin/env bash
set -xeuo pipefail

project_name="lp-reg"
exp_name="Qwen3_7b_lp_reg_reduce_logictoken_penalty_dynamic-$(date +%Y%m%d_%H%M%S)"

LOGS_DIR="${PWD}/logs"
mkdir -p "${LOGS_DIR}"
LOG_FILE="${LOGS_DIR}/${exp_name}.log"

adv_estimator=grpo

loss_mode="lp_reg"
kl_type="low_var_kl"
minp_old_log_prob=True
use_clip=True
minp_p_threshold=0.02
logp_pos_k_percent=0
logp_neg_k_percent=0.01
dynamic_coef=1.0

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=1.0
clip_ratio_high=9.0

max_prompt_length=512
max_response_length=512
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=-1
train_prompt_bsz=256
gen_prompt_bsz=32
train_prompt_mini_bsz=256
n_resp_per_prompt=2
max_token=$((1024 * 4))

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"/share/collab/codemodel/models/Qwen/Qwen2.5-Coder-7B-Instruct"}

CKPTS_DIR=${CKPTS_DIR:-"/nfs_global/S/pengxiong/checkpoint/$project_name/$exp_name"}
TRAIN_FILE=${TRAIN_FILE:-"/nfs_global/S/pengxiong/dataset/Eurus-2-RL-Data/train-code.parquet"}
TEST_FILE=${TEST_FILE:-"/nfs_global/S/pengxiong/dataset/Eurus-2-RL-Data/validation-code.parquet"}
temperature=1.0
top_p=1.0
top_k=-1
ppo_kl_coef=1

sequence_entropy_beta=0.5
sequence_entropy_tau=0.1
sequence_entropy_max_samples=3

use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

logic_aware_lp_reg=True
logic_penalty_weight=${LOGIC_PENALTY_WEIGHT:-0.2}
logic_token_ids=${LOGIC_TOKEN_IDS:-}

if [[ -z "${logic_token_ids}" ]]; then
logic_token_ids=$(MODEL_PATH="${MODEL_PATH}" python3 - <<'PY'
import json
import os
from transformers import AutoTokenizer

model_path = os.environ["MODEL_PATH"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
vocab = tokenizer.get_vocab()
keywords = [
    "if", "for", "while", "break", "continue", "return", "try", "except", "finally",
    "and", "or", "not", "in", "is", "==", "!=", "<", ">", "<=", ">="
]
ids = set()
for token in keywords:
    for candidate in (token, f" {token}", f"Ġ{token}", f"▁{token}"):
        token_id = vocab.get(candidate)
        if token_id is not None:
            ids.add(int(token_id))
print(json.dumps(sorted(ids), separators=(",", ":")))
PY
)
fi

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export REWARD_NUM_PROCESSES=${REWARD_NUM_PROCESSES:-16}
export REWARD_TIMEOUT=${REWARD_TIMEOUT:-60}
export PRIME_CODE_MAX_SAMPLES=${PRIME_CODE_MAX_SAMPLES:-5}
export MPLCONFIGDIR=/nfs_global/S/pengxiong/tmp/matplotlib_config
export SEQUENCE_ENTROPY_OUTPUT_DIR="${PWD}/outputs/sequence_entropy"
mkdir -p "${SEQUENCE_ENTROPY_OUTPUT_DIR}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

HYDRA_FULL_ERROR=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.minp_old_log_prob=${minp_old_log_prob} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_type=${kl_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_clip=${use_clip} \
    actor_rollout_ref.actor.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.logp_pos_k_percent=${logp_pos_k_percent} \
    actor_rollout_ref.actor.logp_neg_k_percent=${logp_neg_k_percent} \
    actor_rollout_ref.actor.dynamic_coef=${dynamic_coef} \
    actor_rollout_ref.actor.minp_p_threshold=${minp_p_threshold} \
    actor_rollout_ref.actor.ppo_kl_coef=${ppo_kl_coef} \
    actor_rollout_ref.actor.logic_aware_lp_reg=${logic_aware_lp_reg} \
    actor_rollout_ref.actor.logic_penalty_weight=${logic_penalty_weight} \
    actor_rollout_ref.actor.logic_token_ids=${logic_token_ids} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.mode=sync \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.use_dynamic_lp_reg=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.sequence_entropy_beta=${sequence_entropy_beta} \
    actor_rollout_ref.rollout.sequence_entropy_tau=${sequence_entropy_tau} \
    actor_rollout_ref.rollout.sequence_entropy_max_samples=${sequence_entropy_max_samples} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=50 \
    trainer.total_epochs=3 \
    trainer.save_train_samples_freq=32 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    2>&1 | tee "${LOG_FILE}"

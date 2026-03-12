import os
import pickle
import keyword

import torch


class CodeLogicAnalyzer:
    def __init__(self, tokenizer, cache_dir="~/.cache/verl/logic_tokens"):
        self.tokenizer = tokenizer
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache_file = os.path.join(self.cache_dir, "python_logic_token_ids.pkl")
        self.logic_keywords = tuple(sorted(set(keyword.kwlist + ["==", "!=", "<", ">", "<=", ">=", "and", "or", "not", "in", "is"])))

    def _candidate_token_forms(self, token: str):
        return (token, f" {token}", f"Ġ{token}", f"▁{token}")

    def _extract_ids_from_vocab(self):
        vocab = self.tokenizer.get_vocab()
        logic_ids = set()
        for token in self.logic_keywords:
            for candidate in self._candidate_token_forms(token):
                token_id = vocab.get(candidate, None)
                if token_id is not None:
                    logic_ids.add(int(token_id))
        return sorted(logic_ids)

    def get_logic_token_ids(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        logic_token_ids = self._extract_ids_from_vocab()
        with open(self.cache_file, "wb") as f:
            pickle.dump(logic_token_ids, f)
        return logic_token_ids

    @staticmethod
    def logic_position_mask(token_ids: torch.Tensor, logic_token_ids: torch.Tensor):
        if logic_token_ids is None or logic_token_ids.numel() == 0:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        return torch.isin(token_ids, logic_token_ids)

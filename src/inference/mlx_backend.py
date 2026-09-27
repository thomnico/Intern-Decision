"""Apple-silicon MLX masked inference (text-only); no XTuner or CUDA imports.

mlx-lm loads the Qwen3.5 language model of the checkpoint and drops the vision tower.
The prompt, skeleton and marker positions are built exactly as in HFBackend; hidden
states are projected to the vocabulary only at the positions before each marker.
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.utils import load
from transformers import AutoTokenizer

from src.inputs.schema import DECISION_TOKEN, compile_row


class MLXBackend:
    def __init__(
        self,
        checkpoint,
        processor_path=None,
        media_root="",
        max_length=8192,
        dtype="bfloat16",
        **kwargs,
    ):
        # `device` and `attn_implementation` are CUDA/HF options; MLX runs on the default Metal device.
        self.checkpoint = str(Path(checkpoint).resolve())
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError("Checkpoint must include its trained decision tokenizer")
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        self.model, _ = load(self.checkpoint)
        if dtype != "bfloat16":
            self.model.set_dtype(getattr(mx, dtype))
        text_model = self.model.language_model
        self.body = text_model.model
        self.head = (
            text_model.model.embed_tokens.as_linear if text_model.args.tie_word_embeddings else text_model.lm_head
        )
        if self.marker_id >= self.body.embed_tokens.weight.shape[0]:
            raise ValueError("Decision marker exceeds checkpoint vocabulary")

    def encode(self, row):
        if row.get("images"):
            raise ValueError("The MLX backend is text-only; use the hf backend for image requests")
        compiled = compile_row(row, include_targets=False)
        # Keep the empty think block and complete assistant skeleton used in training.
        text = self.tokenizer.apply_chat_template(
            compiled.messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            add_vision_id=True,
        )
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_length:
            raise ValueError(f"Example has {len(ids)} tokens, above {self.max_length}; truncation is forbidden")
        positions = [index - 1 for index, token in enumerate(ids) if token == self.marker_id]
        if len(positions) != len(compiled.fields) or min(positions) < 0:
            raise ValueError("Decision marker count or position mismatch")
        return compiled, ids, positions

    def score(self, row):
        compiled, ids, positions = self.encode(row)
        start = time.perf_counter()
        hidden = self.body(mx.array(ids)[None])
        # Project only the marker positions, avoiding L x V logits.
        logits = self.head(hidden[:, mx.array(positions), :])[0].astype(mx.float32)
        mx.eval(logits)
        elapsed = (time.perf_counter() - start) * 1000
        # DecisionEngine decodes a torch tensor; hand over the few selected rows on CPU.
        return compiled, torch.from_numpy(np.array(logits)), len(ids), elapsed

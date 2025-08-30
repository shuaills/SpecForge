# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
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

import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from specforge.modeling._mask_utils import _expand_mask, _make_causal_mask


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """
        pass

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """
        pass

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """
        pass

    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                (batch_size, seq_length),
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=seq_length
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    @abstractmethod
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        The backbone of the draft model.
        """
        pass

    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False
        if hasattr(self, "embedding_proj"):
            self.embedding_proj.weight.requires_grad = False

    def unfreeze_embedding(self):
        """
        Unfreeze the embedding layer for training.
        """
        self.embed_tokens.weight.requires_grad = True
        if hasattr(self, "embedding_proj"):
            self.embedding_proj.weight.requires_grad = True

    @torch.no_grad()
    def load_embedding(
        self,
        model_path: str,
        embedding_key: str = "model.embed_tokens.weight",
        embedding_mode: str = "truncate",
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
            embedding_key (str): Key for the embedding tensor in the model.
            embedding_mode (str): How to handle dimension mismatch:
                - "truncate": Truncate or pad with zeros (default, frozen)
                - "trainable": Initialize randomly and make trainable
                - "projection": Add projection layer from target to draft dimensions
        """
        if os.path.exists(model_path):
            # model_path is a local directory
            # check if there is file ending with index.json
            glob_path = os.path.join(model_path, "*.index.json")
            index_json_path = glob.glob(glob_path)

            if len(index_json_path) == 0:
                # No index.json found, look for single model file
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    with safe_open(safetensors_path, framework="pt") as f:
                        emb_tokens = f.get_tensor(embedding_key)
                        self._handle_embedding_dimension_mismatch(
                            emb_tokens, embedding_mode
                        )
                    return

                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(pytorch_model_path, map_location="cpu")
                    emb_tokens = state_dict[embedding_key]
                    self._handle_embedding_dimension_mismatch(
                        emb_tokens, embedding_mode
                    )
                    return

                raise FileNotFoundError(
                    f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
                )
            if len(index_json_path) > 1:
                raise FileNotFoundError(
                    f"Multiple index.json files found in {model_path}"
                )
            index_json_path = index_json_path[0]

            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]

            if ckpt_file.endswith(".safetensors"):
                with safe_open(
                    os.path.join(model_path, ckpt_file), framework="pt"
                ) as f:
                    emb_tokens = f.get_tensor(embedding_key)
            else:
                state_dict = torch.load(os.path.join(model_path, ckpt_file))
                emb_tokens = state_dict[embedding_key]
            self._handle_embedding_dimension_mismatch(emb_tokens, embedding_mode)
        else:
            # this is the case where model_path is a huggingface repository
            # we first need to locate its local cache
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key, embedding_mode)

    def _handle_embedding_dimension_mismatch(
        self, emb_tokens: torch.Tensor, embedding_mode: str
    ) -> None:
        """
        Handle embedding dimension mismatch with different strategies.

        Args:
            emb_tokens: Target model embeddings
            embedding_mode: Strategy to handle dimension mismatch
        """
        if emb_tokens.shape == self.embed_tokens.weight.shape:
            # Perfect match, just copy
            self.embed_tokens.weight.copy_(emb_tokens)
            return

        vocab_size, target_hidden_size = emb_tokens.shape
        draft_vocab_size, draft_hidden_size = self.embed_tokens.weight.shape

        print(
            f"Embedding dimension mismatch: target {target_hidden_size} -> draft {draft_hidden_size}"
        )
        print(f"Using embedding mode: {embedding_mode}")

        if embedding_mode == "truncate":
            # Original truncate/pad strategy (frozen)
            if draft_hidden_size <= target_hidden_size:
                # Truncate target embeddings to draft size
                emb_tokens_adapted = emb_tokens[:, :draft_hidden_size]
            else:
                # Pad target embeddings with zeros to match draft size
                padding_size = draft_hidden_size - target_hidden_size
                padding = torch.zeros(
                    vocab_size,
                    padding_size,
                    dtype=emb_tokens.dtype,
                    device=emb_tokens.device,
                )
                emb_tokens_adapted = torch.cat([emb_tokens, padding], dim=1)

            self.embed_tokens.weight.copy_(emb_tokens_adapted)
            self.freeze_embedding()

        elif embedding_mode == "trainable":
            # Initialize randomly and make trainable
            print("Initializing embedding weights randomly for training")
            torch.nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
            self.unfreeze_embedding()

        elif embedding_mode == "projection":
            # Add projection layer from target to draft dimensions
            print(
                f"Adding projection layer: {target_hidden_size} -> {draft_hidden_size}"
            )

            # Use target embeddings directly
            if target_hidden_size != draft_hidden_size:
                # Need to resize embedding layer
                import torch.nn as nn

                original_dtype = self.embed_tokens.weight.dtype
                original_device = self.embed_tokens.weight.device
                self.embed_tokens = nn.Embedding(
                    vocab_size,
                    target_hidden_size,
                    padding_idx=self.embed_tokens.padding_idx,
                )
                self.embed_tokens = self.embed_tokens.to(
                    device=original_device, dtype=original_dtype
                )
                self.embed_tokens.weight.copy_(emb_tokens)
                self.freeze_embedding()

                # Add projection layer with same dtype as existing model parameters
                self.embedding_proj = nn.Linear(
                    target_hidden_size, draft_hidden_size, bias=False
                )
                torch.nn.init.normal_(self.embedding_proj.weight, mean=0.0, std=0.02)
                self.embedding_proj.weight.requires_grad = True

                # Move projection layer to same device and dtype as embeddings
                if self.embed_tokens.weight.is_cuda:
                    self.embedding_proj = self.embedding_proj.cuda(
                        self.embed_tokens.weight.device
                    )

                # Ensure projection layer has same dtype as embedding weights
                self.embedding_proj = self.embedding_proj.to(
                    dtype=self.embed_tokens.weight.dtype
                )
            else:
                # Same dimension, just copy
                self.embed_tokens.weight.copy_(emb_tokens)
                self.freeze_embedding()

        else:
            raise ValueError(
                f"Unknown embedding_mode: {embedding_mode}. Use 'truncate', 'trainable', or 'projection'"
            )

    def load_vocab_mapping(self, file_path: str) -> None:
        """
        Load the vocab buffers of the draft model.

        Args:
            file_path (str): The path to the vocab mapping file.
        """
        assert hasattr(self, "t2d") and hasattr(
            self, "d2t"
        ), "t2d and d2t buffersare not found in the draft model, please check your draft model implementation"
        vocab_mapping = torch.load(file_path)
        self.t2d.copy_(vocab_mapping["t2d"])
        self.d2t.copy_(vocab_mapping["d2t"])

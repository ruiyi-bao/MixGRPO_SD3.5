# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import json
import os
import random


class LatentDataset(Dataset):
    """
    SD3.5 Dataset for loading pre-computed embeddings
    Note: text_ids is delected (unlike Flux)
    """
    def __init__(
        self, json_path, num_latent_t, cfg_rate,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        #self.video_dir = os.path.join(self.datase_dir_path, "video")
        #self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path, "pooled_prompt_embeds"
        )
        # SD3.5 doesn't use text_ids, removed text_ids_dir
        
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096] for CFG
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        #latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        pooled_prompt_embeds_file = self.data_anno[idx]["pooled_prompt_embeds_path"]
        # SD3.5 doesn't use text_ids
        
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            pooled_prompt_embeds = torch.zeros_like(torch.load(
                os.path.join(
                    self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file
                ),
                map_location="cpu",
                weights_only=True,
            ))
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            pooled_prompt_embeds = torch.load(
                os.path.join(
                    self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file
                ),
                map_location="cpu",
                weights_only=True,
            )
        
        # Remove extra dimensions if present
        # Expected shapes: prompt_embed [seq_len, hidden_size], pooled_prompt_embeds [hidden_size]
        # If shapes are [1, seq_len, hidden_size] or [1, hidden_size], squeeze the first dimension
        if prompt_embed.ndim == 3 and prompt_embed.shape[0] == 1:
            prompt_embed = prompt_embed.squeeze(0)
        if pooled_prompt_embeds.ndim == 2 and pooled_prompt_embeds.shape[0] == 1:
            pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
        
        return prompt_embed, pooled_prompt_embeds, self.data_anno[idx]['caption']

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    """
    Collate function for SD3.5 dataset
    Returns: (prompt_embeds, pooled_prompt_embeds, captions)
    Note: text_ids is delected (unlike Flux which needs it)
    """
    prompt_embeds, pooled_prompt_embeds, captions = zip(*batch)

    # Stack embeddings
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    
    return prompt_embeds, pooled_prompt_embeds, captions


# if __name__ == "__main__":
#     dataset = LatentDataset("data/rl_embeddings/videos2caption.json", num_latent_t=28, cfg_rate=0.0)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=2, shuffle=False, collate_fn=latent_collate_function
#     )
#     for prompt_embed, prompt_attention_mask, caption in dataloader:
#         print(
#             prompt_embed.shape,
#             prompt_attention_mask.shape,
#             caption
#         )
#         import pdb

#         pdb.set_trace()
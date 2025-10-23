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

import argparse
import math
import os
from pathlib import Path

import wandb.util
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
# from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers import FlowMatchEulerDiscreteScheduler
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import SD3Transformer2DModel, AutoencoderKL, StableDiffusion3Pipeline
from fastvideo.pipelines import setup_sd3_pipeline_for_grpo
from fastvideo.utils.grpo_states import GRPOTrainingStates
import json
from fastvideo.models.reward_model.image_reward import ImageRewardModel
from fastvideo.models.reward_model.pick_score import PickScoreRewardModel
from fastvideo.models.reward_model.unified_reward import UnifiedRewardModel
from fastvideo.models.reward_model.hps_score import HPSClipRewardModel
from fastvideo.models.reward_model.clip_score import CLIPScoreRewardModel
from fastvideo.models.reward_model.utils import compute_reward, balance_pos_neg
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional
import concurrent.futures
from fastvideo.utils.sampling_utils import flow_grpo_step, dance_grpo_step, run_sample_step, sd3_time_shift, dpm_step

def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def grpo_one_step(
    args,
    latents,
    pre_latents,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    transformer,
    timesteps,
    i,
    sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    if args.dpm_algorithm_type == "null" or ("dpmsolver" in args.dpm_algorithm_type and args.dpm_apply_strategy == "post"):
        if args.flow_grpo_sampling:
            z, pred_original, log_prob, prev_latents_mean, std_dev_t = flow_grpo_step(
                model_output=pred,
                latents=latents.to(torch.float32),
                eta=args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=pre_latents.to(torch.float32),
                determistic=False,
            )
        else:
            z, pred_original, log_prob = dance_grpo_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    elif "dpmsolver" in args.dpm_algorithm_type:
        z, pred_original, log_prob = dpm_step(
            args,
            model_output=pred,
            sample=latents.to(torch.float32),
            step_index=i,
            timesteps=sigma_schedule[:-1],
            dpm_state=None,
            generator=torch.Generator(device=latents.device),
            sde_solver=True,
            sigmas=sigma_schedule,
        )
    return log_prob

def sample_reference_model(
    args,
    device,
    pipeline,  # Now takes pipeline instead of separate transformer/vae
    encoder_hidden_states,
    pooled_prompt_embeds,
    reward_models,
    caption,
    timesteps_train, # index
    global_step,
    reward_weights,
    neg_prompt_embed=None,
    neg_pooled_embed=None,
):
    """Sample using Flow-GRPO pipeline approach for high-quality images"""
    w, h = args.w, args.h
    sample_steps = args.sampling_steps

    # Get scheduler info for compatibility with training loop
    scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    scheduler.set_timesteps(args.sampling_steps, device=device)
    timesteps = scheduler.timesteps.to(device)
    sigma_schedule = scheduler.sigmas.to(device)

    B = encoder_hidden_states.shape[0]
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []
    all_multi_rewards = {}

    rank = int(os.environ["RANK"])
    image_dir = f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}/images"
    os.makedirs(image_dir, exist_ok=True)

    if dist.get_rank() == 0:
        sampling_time = 0

    for index, batch_idx in enumerate(batch_indices):
        if dist.get_rank() == 0:
            meta_sampling_time = time.time()

        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]

        # Use text prompts directly instead of pre-encoded embeddings for better quality
        # This matches the standalone test that generated perfect images
        with torch.no_grad():
            images, batch_latents_list, batch_log_probs_list = pipeline.pipeline_with_logprob(
                prompt=batch_caption[0],  # Use text prompt directly
                num_inference_steps=sample_steps,
                guidance_scale=args.cfg if args.cfg > 1.0 else 0.0,
                output_type="pil",
                height=h,
                width=w,
            )

        if dist.get_rank() == 0:
            sampling_time += time.time() - meta_sampling_time
            main_print(f"##### Sampling time per data: {sampling_time/(index+1)} seconds")

        # Convert list of latents to tensor: batch_latents_list is [latent0, latent1, ..., latentN]
        # Each latent is shape (B, C, H, W), stack to get (num_steps+1, B, C, H, W)
        # Then permute to (B, num_steps+1, C, H, W) to match training expectations
        batch_latents = torch.stack(batch_latents_list, dim=0).permute(1, 0, 2, 3, 4)  # (T+1, B, C, H, W) -> (B, T+1, C, H, W)
        if batch_log_probs_list:
            batch_log_probs = torch.stack(batch_log_probs_list, dim=0).permute(1, 0)  # (T, B) -> (B, T)
        else:
            batch_log_probs = torch.zeros(len(batch_idx), len(batch_latents_list)-1, device=device)

        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        # Save first image from this batch
        if index == 0:
            images[0].save(f"{image_dir}/sd35_{global_step}_{rank}_final.png")

        # Compute rewards using the decoded images
        with torch.no_grad():
            prompts = [batch_caption[0]]
            rewards, successes, rewards_dict, successes_dict = compute_reward(
                images[:1],  # Just the first image
                prompts,
                reward_models,
                reward_weights,
            )
            if args.multi_reward_mix == "reward_aggr":
                all_rewards.append(torch.tensor(rewards, device=device, dtype=torch.float32))
            elif args.multi_reward_mix == "advantage_aggr":
                for model_name, model_rewards in rewards_dict.items():
                    if model_name not in all_multi_rewards:
                        all_multi_rewards[model_name] = {"rewards": [], "successes": []}
                    all_multi_rewards[model_name]["rewards"].append(
                        torch.tensor(model_rewards, device=device, dtype=torch.float32)
                    )
                    all_multi_rewards[model_name]["successes"].append(
                        torch.tensor(successes_dict[model_name], device=device, dtype=torch.float32)
                    )

    # Concatenate all results
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    if args.multi_reward_mix == "reward_aggr":
        all_rewards_res = torch.cat(all_rewards, dim=0)
    elif args.multi_reward_mix == "advantage_aggr":
        all_rewards_res = {}
        for model_name, model_rewards in all_multi_rewards.items():
            all_rewards_res[model_name] = torch.cat(model_rewards["rewards"], dim=0)

    return all_rewards_res, all_latents, all_log_probs, timesteps, sigma_schedule


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    transformer,
    reference_pipeline,  # Pipeline for sampling, separate from training transformer
    reward_models,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    timesteps_train, # index
    global_step,
    reward_weights,
    neg_prompt_embed=None,
    neg_pooled_embed=None,
):
    total_loss = 0.0
    kl_total_loss = 0.0
    policy_total_loss = 0.0
    total_clip_frac = 0.0
    optimizer.zero_grad()
    (
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        caption,
    ) = next(loader)
    #device = latents.device
    #按群采样次数复制输入prompt（12次）
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)


        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    # Use reference pipeline for high-quality sampling
    reward, all_latents, all_log_probs, timesteps_from_scheduler, sigma_schedule = sample_reference_model(
            args,
            device,
            reference_pipeline,
            encoder_hidden_states,
            pooled_prompt_embeds,
            reward_models,
            caption,
            timesteps_train,
            global_step,
            reward_weights,
            neg_prompt_embed=neg_prompt_embed if args.cfg > 1.0 else None,
            neg_pooled_embed=neg_pooled_embed if args.cfg > 1.0 else None,
        )
    batch_size = all_latents.shape[0]
    # Use scheduler timesteps directly instead of int(sigma * 1000)
    timesteps = timesteps_from_scheduler.unsqueeze(0).expand(batch_size, -1)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }

    if args.multi_reward_mix == "advantage_aggr":
        samples["rewards"] = {}
        gathered_reward = {}
        for model_name, model_rewards in reward.items():
            gathered_reward[model_name] = gather_tensor(model_rewards.to(torch.float32))
            samples["rewards"][model_name] = model_rewards.to(torch.float32)
    elif args.multi_reward_mix == "reward_aggr":
        samples["rewards"] = reward.to(torch.float32)
        gathered_reward = gather_tensor(samples["rewards"])

    if dist.get_rank()==0:
        print(f"gathered_{args.reward_model}", gathered_reward)
        reward_dir = f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}"
        with open(f'{reward_dir}/sd35_{args.reward_model}_{args.training_strategy}_{args.experiment_name}.txt', 'a') as f: 
            if args.multi_reward_mix == "advantage_aggr":
                for model_name, model_rewards in gathered_reward.items():
                    f.write(f"{model_name}: {model_rewards.mean().item()}\n")
            elif args.multi_reward_mix == "reward_aggr":
                f.write(f"reward: {gathered_reward.mean().item()}\n")

    #计算advantage
    if args.use_group:
        if args.multi_reward_mix == "advantage_aggr":
            model_advantages = {}
            for model_name, model_rewards in samples["rewards"].items():
                n = len(model_rewards) // (args.num_generations)
                advantages = torch.zeros_like(model_rewards)

                for i in range(n):
                    start_idx = i * args.num_generations
                    end_idx = (i + 1) * args.num_generations
                    group_rewards = model_rewards[start_idx:end_idx]
                    if args.trimmed_ratio > 0:
                        sorted_rewards = torch.sort(group_rewards)[0]
                        len_sorted_rewards = len(sorted_rewards)
                        trim_size = min(int(len_sorted_rewards * args.trimmed_ratio), len_sorted_rewards - 1)
                        trimmed_rewards = sorted_rewards[trim_size:]
                        group_mean = trimmed_rewards.mean()
                        group_std = trimmed_rewards.std() + 1e-8
                    else:
                        group_mean = group_rewards.mean()
                        group_std = group_rewards.std() + 1e-8
                    advantages[start_idx:end_idx] += (group_rewards - group_mean) / group_std
                
                model_advantages[model_name] = advantages
            # add all advantages
            merged_advantages = torch.zeros_like(samples["rewards"][list(samples["rewards"].keys())[0]])
            for model_name, model_advs in model_advantages.items():
                merged_advantages += model_advs * reward_weights[model_name]
            samples["advantages"] = merged_advantages
        
        elif args.multi_reward_mix == "reward_aggr":
            n = len(samples["rewards"]) // (args.num_generations)
            advantages = torch.zeros_like(samples["rewards"])
            
            for i in range(n):
                start_idx = i * args.num_generations
                end_idx = (i + 1) * args.num_generations
                group_rewards = samples["rewards"][start_idx:end_idx]
                if args.trimmed_ratio > 0:
                    sorted_rewards = torch.sort(group_rewards)[0]
                    len_sorted_rewards = len(sorted_rewards)
                    trim_size = min(int(len_sorted_rewards * args.trimmed_ratio), len_sorted_rewards - 1)
                    trimmed_rewards = sorted_rewards[trim_size:]
                    group_mean = trimmed_rewards.mean()
                    group_std = trimmed_rewards.std() + 1e-8
                else:
                    group_mean = group_rewards.mean()
                    group_std = group_rewards.std() + 1e-8

                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            
            samples["advantages"] = advantages
        else:
            raise ValueError(
                f"multi_reward_mix {args.multi_reward_mix} is not supported."
            )
    else:
        if args.multi_reward_mix == "advantage_aggr":
            raise ValueError(
                "multi_reward_mix 'advantage_aggr' is not supported when use_group is False."
            )
        elif args.multi_reward_mix == "reward_aggr":
            # Use unbiased=False to avoid NaN when there's only 1 sample
            reward_std = gathered_reward.std(unbiased=False)
            # Handle case when std is 0 (all rewards are the same)
            if reward_std < 1e-8 or torch.isnan(reward_std):
                advantages = torch.zeros_like(samples["rewards"])
            else:
                advantages = (samples["rewards"] - gathered_reward.mean())/(reward_std + 1e-8)
            samples["advantages"] = advantages
        else:
            raise ValueError(
                f"multi_reward_mix {args.multi_reward_mix} is not supported."
            )

    if args.training_strategy == "all":
        perms = torch.stack(
            [
                torch.randperm(len(samples["timesteps"][0]))
                for _ in range(batch_size)
            ]
        ).to(device) 
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size).to(device) [:, None],
                perms,
            ]

    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
        if k != "rewards"
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    if args.training_strategy == "part":
        train_timesteps = timesteps_train
    elif args.training_strategy == "all":
        if args.frozen_init_timesteps > 0:
            assert args.frozen_init_timesteps <= len(samples["timesteps"][0])
            train_timesteps = range(args.frozen_init_timesteps)
        else:
            train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
            train_timesteps = range(train_timesteps)
    
    if args.training_strategy == "part":
        if args.advantage_rerange_strategy == "null":
            pass
        elif args.advantage_rerange_strategy == "random":
            samples_batched_list = balance_pos_neg(samples_batched_list, use_random=True)
        elif args.advantage_rerange_strategy == "balance":
            samples_batched_list = balance_pos_neg(samples_batched_list, use_random=False)
        else:
            raise ValueError(
                f"advantage_rerange_strategy {args.advantage_rerange_strategy} is not supported."
            )
    if dist.get_rank() == 0:
        optimize_sampling_time = 0

    grad_norm = torch.tensor(0.0)  # Initialize grad_norm

    for i,sample in list(enumerate(samples_batched_list)):
        for _ in train_timesteps:
            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time()
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_] if args.training_strategy == "all" else _,
                sigma_schedule,
            )

            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time() - meta_optimize_sampling_time
                optimize_sampling_time += meta_optimize_sampling_time

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * len(train_timesteps))
            kl_loss = 0.5 * torch.mean((new_log_probs - sample["log_probs"][:,_]) ** 2) / (args.gradient_accumulation_steps * len(train_timesteps))
            loss = policy_loss + args.kl_coeff * kl_loss

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

            avg_policy_loss = policy_loss.detach().clone()
            dist.all_reduce(avg_policy_loss, op=dist.ReduceOp.AVG)
            policy_total_loss += avg_policy_loss.item()

            avg_kl_loss = kl_loss.detach().clone()
            dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.AVG)
            kl_total_loss += avg_kl_loss.item()

            avg_clip_frac = clip_frac.detach().clone()
            dist.all_reduce(avg_clip_frac, op=dist.ReduceOp.AVG)
            total_clip_frac += avg_clip_frac.item()

        if dist.get_rank() == 0:
            main_print(f"##### Optimize sampling time per step: {optimize_sampling_time / (i+1)} seconds")

        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
            print("kl loss", kl_loss.item())
        dist.barrier()

    if args.multi_reward_mix == "advantage_aggr":
        gathered_reward_res = {}
        for model_name, model_rewards in gathered_reward.items():
            gathered_reward_res[model_name] = model_rewards.mean().item()
    elif args.multi_reward_mix == "reward_aggr":
        gathered_reward_res = gathered_reward.mean().item()

    return total_loss, grad_norm.item(), policy_total_loss, kl_total_loss, total_clip_frac, gathered_reward_res


def main(args):
    ############################# Init #############################
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}", exist_ok=True)
        args_dict = vars(args)
        run_id = wandb.util.generate_id()
        args_dict["wandb_id"] = run_id
        with open(f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    
    ############################# Build reward models #############################
    reward_models = []
    if args.reward_model == "hpsv2":
        reward_models.append(HPSClipRewardModel(
            device=device,
            clip_ckpt_path=args.hps_clip_path,
            hps_ckpt_path=args.hps_path,
        ))
    elif args.reward_model == "image_reward":
        reward_models.append(ImageRewardModel(
            model_name=args.image_reward_path,
            device=device,
            med_config=args.image_reward_med_config,
            http_proxy=args.image_reward_http_proxy,
            https_proxy=args.image_reward_https_proxy,
        ))

    elif args.reward_model == "clip_score":
        reward_models.append(CLIPScoreRewardModel(
            clip_model_path=args.clip_score_path,
            device=device,
        ))
    elif args.reward_model == "pick_score":
        reward_models.append(PickScoreRewardModel(
            device=device,
            http_proxy=args.pick_score_http_proxy,
            https_proxy=args.pick_score_https_proxy,
        ))
    elif args.reward_model == "unified_reward":
        unified_reward_urls = args.unified_reward_url.split(",")
        
        if isinstance(unified_reward_urls, list):
            num_urls = len(unified_reward_urls)
            ur_url_idx = rank % num_urls
            ur_url = unified_reward_urls[ur_url_idx]
            print(f"Rank {rank} using unified-reward URL: {ur_url}")
        reward_models.append(UnifiedRewardModel(
            api_url=ur_url,
            default_question_type=args.unified_reward_default_question_type,
            num_workers=args.unified_reward_num_workers,
        ))
    
    elif args.reward_model == "hpsv2_clip_score":
        reward_models.append(HPSClipRewardModel(
            device=device,
            clip_ckpt_path=args.hps_clip_path,
            hps_ckpt_path=args.hps_path,
        ))
        reward_models.append(CLIPScoreRewardModel(
            clip_model_path=args.clip_score_path,
            device=device,
        ))
    elif args.reward_model == "multi_reward":
        reward_models.append(HPSClipRewardModel(
            device=device,
            clip_ckpt_path=args.hps_clip_path,
            hps_ckpt_path=args.hps_path,
        ))
        reward_models.append(ImageRewardModel(
            model_name=args.image_reward_path,
            device=device,
            med_config=args.image_reward_med_config,
            http_proxy=args.image_reward_http_proxy,
            https_proxy=args.image_reward_https_proxy,
        ))
        reward_models.append(PickScoreRewardModel(
            device=device,
            http_proxy=args.pick_score_http_proxy,
            https_proxy=args.pick_score_https_proxy,
        ))
        if args.unified_reward_url is not None:
            unified_reward_urls = args.unified_reward_url.split(",")
            if isinstance(unified_reward_urls, list):
                num_urls = len(unified_reward_urls)
                ur_url_idx = rank % num_urls
                ur_url = unified_reward_urls[ur_url_idx]
                print(f"Rank {rank} using unified-reward URL: {ur_url}")
            reward_models.append(UnifiedRewardModel(
                api_url=ur_url,
                default_question_type=args.unified_reward_default_question_type,
                num_workers=args.unified_reward_num_workers,
            ))
    else:
        raise ValueError(f"Unsupported reward model: {args.reward_model}")


    ############################# Reward Models Setting #############################
        
    # Initialize reward model weights only for activated models
    reward_weights = {}
    for model in reward_models:
        model_name = type(model).__name__    
        if model_name == 'HPSClipRewardModel':
            weight = args.hps_weight
        elif model_name == 'CLIPScoreRewardModel':
            weight = args.clip_score_weight
        elif model_name == 'ImageRewardModel':
            weight = args.image_reward_weight
        elif model_name == 'UnifiedRewardModel':
            weight = args.unified_reward_weight
        elif model_name == 'PickScoreRewardModel':
            weight = args.pick_score_weight
        else:
            weight = 1.0
        reward_weights[model_name] = weight

    # Normalize weights
    total_weight = sum(reward_weights.values())
    if total_weight > 0:
        reward_weights = {k: v/total_weight for k, v in reward_weights.items()}
    else:
        print("No reward models activated or all weights are 0!")
        reward_weights = {type(model).__name__: 1.0/len(reward_models) for model in reward_models}

    print(f"reward_weights: {reward_weights}")

    ############################# Build SD3.5 #############################
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32

    transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
    )

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    # Load reference pipeline for high-quality sampling (Flow-GRPO approach)
    main_print(f"--> Loading reference pipeline for sampling...")
    reference_pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    reference_pipeline = reference_pipeline.to(device)
    reference_pipeline = setup_sd3_pipeline_for_grpo(reference_pipeline)
    main_print(f"--> Reference pipeline loaded and ready")

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    #不知道要不要改
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else args.seed if args.seed is not None else 42
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=sampler_seed
        )


    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Load negative embeddings for CFG if cfg > 1.0
    if args.cfg > 1.0:
        neg_embed_dir = os.path.dirname(args.data_json_path)
        neg_prompt_embed_path = os.path.join(neg_embed_dir, "neg_prompt_embed.pt")
        neg_pooled_embed_path = os.path.join(neg_embed_dir, "neg_pooled_prompt_embed.pt")

        neg_prompt_embed = torch.load(neg_prompt_embed_path, map_location=f"cuda:{device}").unsqueeze(0)  # Add batch dim
        neg_pooled_embed = torch.load(neg_pooled_embed_path, map_location=f"cuda:{device}").unsqueeze(0)
        main_print(f"Loaded negative embeddings for CFG (guidance_scale={args.cfg})")
    else:
        neg_prompt_embed = None
        neg_pooled_embed = None

    #vae.enable_tiling()

    if rank <= 0:
        project = "sd3.5"
        wandb_run = wandb.init(
            project=project, 
            config=args, 
            name=args.experiment_name,
            id=run_id,
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    if args.training_strategy == "part":
        grpo_states = GRPOTrainingStates(
            iters_per_group=args.iters_per_group,
            group_size=args.group_size,
            max_timesteps=args.sampling_steps-2,  # Because the max timestep index is args.sampling_steps - 2
            cur_timestep=0,
            cur_iter_in_group=0,
            sample_strategy=args.sample_strategy,
            prog_overlap=args.prog_overlap,
            prog_overlap_step=args.prog_overlap_step,
            max_iters_per_group=args.max_iters_per_group,
            min_iters_per_group=args.min_iters_per_group,
            roll_back=args.roll_back,
        )

    global_step = -1
    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        
        for step in range(init_steps+1, args.max_train_steps+1):
            global_step += 1
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, f"{args.output_dir}/{args.training_strategy}_{args.experiment_name}",
                                step, epoch)

                dist.barrier()

            if args.training_strategy == "part":
                timesteps_train = grpo_states.get_current_timesteps()
                grpo_states.update_iteration()
            elif args.training_strategy == "all":
                timesteps_train = [ti for ti in range(args.sampling_steps)]

            loss, grad_norm, policy_loss, kl_loss, clip_frac, reward = train_one_step(
                args,
                device,
                transformer,
                reference_pipeline,
                reward_models,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                timesteps_train,
                global_step,
                reward_weights,
                neg_prompt_embed,
                neg_pooled_embed,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                log_dict = {
                    "train_loss": loss,
                    "policy_loss": policy_loss,
                    "kl_loss": kl_loss,
                    "clip_frac": clip_frac,
                    "cur_timesteps": grpo_states.cur_timestep if args.training_strategy == "part" else 0,
                    "cur_iter_in_group": grpo_states.cur_iter_in_group if args.training_strategy == "part" else 0,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                }
                if args.multi_reward_mix == "advantage_aggr":
                    for model_name, model_rewards in reward.items():
                        log_dict[f"reward_{model_name}"] = model_rewards
                elif args.multi_reward_mix == "reward_aggr":
                    log_dict["reward"] = reward

                wandb.log(log_dict, step=global_step)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_intermediate_steps",
        type=str,
        default=None,
        help=(
            "Comma-separated list of denoising steps to save as images (e.g., '1,5,10,15,20'). "
            "Useful for debugging and visualizing the denoising process during training. "
            "If not specified, only the final image is saved."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="noise eta (default: 1.0 for stochastic sampling)",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--advantage_rerange_strategy",
        type=str,
        default="null",
        choices=["random", "balance", "null"],
        help="Rerange strategy for advantages when computing loss"
    )

    #################### MixGRPO ####################
    parser.add_argument(
        "--flow_grpo_sampling",
        action="store_true",
        default=False,
        help="whether to use flow grpo sampling, True for MixGRPO, False for DanceGRPO",
    )
    parser.add_argument(
        "--drop_last_sample",
        action="store_true",
        default=False,
        help="whether to drop the last sample in the batch if it is not complete, True for DanceGRPO but False for MixGRPO",
    )
    parser.add_argument(
        "--trimmed_ratio",
        type=float,
        default=0.0,
        help="ratio of trimmed for advantage computation, now is no used",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="test",
        help="experiment name, used for saving images and logs",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="all",
        choices=["part", "all"],
        help="training strategy, part means MixGRPO, all means DanceGRPO",
    )
    parser.add_argument(
        "--frozen_init_timesteps",
        type=int,
        default=-1,
        help="when training_strategy is 'all' and frozen_init_timesteps >0, it is used for freezing timesteps"
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.01,
        help="coefficient for KL loss",
    )
    
    # Sliding Window
    parser.add_argument(
        "--iters_per_group",
        type=int,
        default=25,
        help="shift interval, moving the window after iters_per_group iterations",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="sliding window size",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="progressive",
        choices=["progressive", "random", "decay", "exp_decay"],
        help="Scheduling policy for optimized timesteps",
    )
    parser.add_argument(
        "--prog_overlap",
        action="store_true",
        default=False,
        help="Whether to overlap when moving the window"
    )
    parser.add_argument(
        "--prog_overlap_step",
        type=int,
        default=1,
        help="the window stride when prog_overlap is True",
    )
    parser.add_argument(
        "--max_iters_per_group",
        type=int,
        default=10,
        help="maximum shift interval in 'decay' strategy",
    )
    parser.add_argument(
        "--min_iters_per_group",
        type=int,
        default=1,
        help="minimum shift interval in 'decay' strategy",
    )
    parser.add_argument(
        "--roll_back",
        action="store_true",
        default=False,
        help="whether to roll back (restart) the sliding window",
    )
    #################### Reward ####################
    parser.add_argument(
        "--reward_model",
        type=str,
        default="hpsv2",
        choices=["hpsv2", "clip_score", "image_reward", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"],
        help="reward model to use"
    )
    parser.add_argument(
        "--hps_path",
        type=str,
        default="hps_ckpt/HPS_v2.1_compressed.pt",
        help="path to load hps reward model",
    )
    parser.add_argument(
        "--hps_clip_path",
        type=str,
        default="hps_ckpt/open_clip_pytorch_model.bin",
        help="path to load hps clip model",
    )
    parser.add_argument(
        "--clip_score_path",
        type=str,
        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
        help="clip model type"
    )
    parser.add_argument(
        "--image_reward_path",
        type=str,
        default="./image_reward_ckpt/ImageReward.pt",
        help="path to load image reward model",
    )
    parser.add_argument(
        "--image_reward_med_config",
        type=str,
        default="./image_reward_ckpt/med_config.json",
        help="path to load image reward model med config",
    )
    parser.add_argument(
        "--image_reward_http_proxy",
        type=str,
        default=None,
        help="http proxy for image reward model",
    )
    parser.add_argument(
        "--image_reward_https_proxy",
        type=str,
        default=None,
        help="https proxy for image reward model",
    )
    parser.add_argument(
        "--pick_score_http_proxy",
        type=str,
        default=None,
        help="http proxy for pick score reward model",
    )
    parser.add_argument(
        "--pick_score_https_proxy",
        type=str,
        default=None,
        help="https proxy for pick score reward model",
    )
    parser.add_argument(
        "--unified_reward_url",
        type=str,
        default=None,
        help="API URL for the unified reward model",
    )
    parser.add_argument(
        "--unified_reward_default_question_type",
        type=str,
        default=None,
        help="Default question type for the unified reward model",
    )
    parser.add_argument(
        "--unified_reward_num_workers",
        type=int,
        default=1,
        help="Number of workers for the unified reward model",
    )
    parser.add_argument(
        "--multi_reward_mix",
        type=str,
        default="reward_aggr",
        choices=["advantage_aggr", "reward_aggr"],
        help="How to mix multiple rewards (use 'reward_aggr' when use_group is False)",
    )
    parser.add_argument(
        "--hps_weight",
        type=float,
        default=1.0,
        help="weight for hps reward model",
    )
    parser.add_argument(
        "--clip_score_weight",
        type=float,
        default=1.0,
        help="weight for clip score reward model",
    )
    parser.add_argument(
        "--image_reward_weight",
        type=float,
        default=1.0,
        help="weight for image reward model",
    )
    parser.add_argument(
        "--pick_score_weight",
        type=float,
        default=1.0,
        help="weight for pick score reward model",
    )
    parser.add_argument(
        "--unified_reward_weight",
        type=float,
        default=1.0,
        help="weight for unified reward model",
    )

    #################### Sampling ####################
    parser.add_argument(
        "--dpm_algorithm_type",
        type=str,
        default="null",
        choices=["null", "dpmsolver", "dpmsolver++"],
        help="null means no DPM-Solver, dpmsolver means DPM-Solver, dpmsolver++ means DPM-Solver++",
    )
    parser.add_argument(
        "--dpm_apply_strategy",
        type=str,
        default="post",
        choices=["post", "all"],
        help="post means apply DPM-Solver the ODE sampling process after SDE, all means apply DPM-Solver to all timesteps",
    )
    parser.add_argument(
        "--dpm_post_compress_ratio",
        type=float,
        default=0.4,
        help="when dpm_apply_strategy is post, the timesteps for ODE aftet SDE is compressed by this ratio",
    )
    parser.add_argument(
        "--dpm_solver_order",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Order of the DPM-Solver method. 1 is default DDIM (FM Sampling)",
    )
    parser.add_argument(
       "--dpm_solver_type",
        type=str,
        default="heun",
        choices=["heun", "midpoint"],
        help="when dpm_solver_order is 2, the type of DPM-Solver method.",
    )

    #################### Wandb ####################
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help="Wandb API key for logging. If not provided, will not log to Wandb.",
    )

    args = parser.parse_args()
    wandb.login(key=args.wandb_key, relogin=True)

    if args.image_reward_http_proxy == "None":
        args.image_reward_http_proxy = None
    if args.image_reward_https_proxy == "None":
        args.image_reward_https_proxy = None
    if args.pick_score_http_proxy == "None":
        args.pick_score_http_proxy = None
    if args.pick_score_https_proxy == "None":
        args.pick_score_https_proxy = None
    if args.unified_reward_url == "None":
        args.unified_reward_url = None

    main(args)

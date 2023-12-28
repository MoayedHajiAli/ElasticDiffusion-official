from typing import Any

from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor)
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid 
import os
import math
import numpy as np

import time
from contextlib import contextmanager
from fractions import Fraction
import hashlib



class TimeIt:
    def __init__(self, sync_gpu=False):
        self.sync_gpu = sync_gpu
        self.total_time = {}

    def time_function(self, func):
        def wrapper(*args, **kwargs):
            if self.sync_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            result = func(*args, **kwargs)

            if self.sync_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
                
            self.total_time[f'FUNCTION_{func.__name__}'] = self.total_time.get(f'FUNCTION_{func.__name__}', 0) + (end_time - start_time)
            return result
        return wrapper
    

    @contextmanager
    def time_block(self, block_title):
        if self.sync_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        try:
            yield
        finally:
            if self.sync_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            self.total_time[f'BLOCK_{block_title}'] = self.total_time.get(f'BLOCK_{block_title}', 0) + (end_time - start_time)
    
    def print_results(self):
        for key, time_spent in self.total_time.items():
            print(f"{key} took total {time_spent} seconds to complete.")


class LinearScheduler():
    def __init__(self, steps, start_val, stop_val):
        self.steps = steps
        self.start_val = start_val
        self.stop_val = stop_val
    
    def __call__(self, t, *args: Any, **kwds: Any) -> Any:
        if t >= self.steps:
            return self.stop_val
        return self.start_val + (self.stop_val - self.start_val) / self.steps * t


class ConstScheduler():
    def __init__(self, steps, start_val, stop_val):
        self.steps = steps
        self.start_val = start_val
        self.stop_val = stop_val
    
    def __call__(self, t, *args: Any, **kwds: Any) -> Any:
        if t >= self.steps:
            return self.stop_val
        return self.start_val

class CosineScheduler():
    def __init__(self, steps, cosine_scale, factor=0.01):
            self.steps = steps
            self.cosine_scale = cosine_scale
            self.factor = factor
    
    def __call__(self, t, *args: Any, **kwds: Any) -> Any:
        if t >= self.steps:
            return 0
        
        cosine_factor = 0.5 * (1 + np.cos(np.pi * t / self.steps))
        return self.factor * (cosine_factor ** self.cosine_scale)

timelog = TimeIt(sync_gpu=False) 
class ElasticDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', 
                 verbose=False,
                 log_freq=5,
                 view_batch_size=1,
                 torch_dtype=torch.float32):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.verbose = verbose
        self.torch_dtype = torch_dtype
        self.view_batch_size = view_batch_size
        self.log_freq = log_freq

        print(f'[INFO] loading stable diffusion...')
        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        
        elif self.sd_version == 'XL1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            print(f'[INFO] using hugging face custom model key: {self.sd_version}')
            model_key = self.sd_version

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch_dtype).to(self.device)
        self.tokenizer = [CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", torch_dtype=torch_dtype)]
        self.text_encoder = [CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch_dtype).to(self.device)]
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch_dtype).to(self.device)

        if self.sd_version == 'XL1.0':
            self.text_encoder.append(CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch_dtype).to(self.device))
            self.tokenizer.append(CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2", torch_dtype=torch_dtype))

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.requires_grad(self.vae, False)
        self.set_view_config()

        print(f'[INFO] loaded stable diffusion!')
    
    def set_view_config(self, stride=None):
        if self.sd_version == 'XL1.0':
            self.view_config = {
            "window_size": stride if stride is not None else 64,
            "stride": stride if stride is not None else 64}
            self.view_config["context_size"] = 128 - self.view_config["window_size"]
        else:
            self.view_config = {
            "window_size": stride if stride is not None else 32,
            "stride": stride if stride is not None else 32}
            self.view_config["context_size"] = 64 - self.view_config["window_size"]

    def seed_everything(self, seed, seed_np=True):
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        
        if seed_np:
            np.random.seed(seed)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def get_views(self, panorama_height, panorama_width, h_ws=64, w_ws=64, stride=32, **kwargs):
        
        if int(panorama_height / 8) != panorama_height/ 8 or int(panorama_width / 8) != panorama_width / 8:
            raise f"height {panorama_height} and Width {panorama_width} must be divisable by 8"

        panorama_height //= 8 # go to LDM latent size
        panorama_width //= 8

        num_blocks_height = math.ceil((panorama_height - h_ws) / stride) + 1 if stride else 1
        num_blocks_width = math.ceil((panorama_width - w_ws) / stride) + 1 if stride else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)

        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + h_ws
            if h_end > panorama_height: # adjust last crop
                h_start -= h_end - panorama_height
                h_end = panorama_height
                h_start = max(0, h_start)

            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + w_ws

            if w_end > panorama_width: # adjust last crop
                w_start -= w_end - panorama_width
                w_end = panorama_width
                w_start = max(0, w_start)

            views.append((h_start, h_end, w_start, w_end))
    
        return views
    
    ## Copied from https://github.com/huggingface/diffusers/blob/cf03f5b7188c603ff037d686f7256d0571fbd651/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L94
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder[1].config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def encoder_prompt(self, prompt, encoder_id):
        text_input = self.tokenizer[encoder_id](prompt, padding='max_length', max_length=self.tokenizer[encoder_id].model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder[encoder_id](text_input.input_ids.to(self.device), output_hidden_states=True)
        return text_embeddings

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        if self.sd_version == 'XL1.0':
            text_embeddings = torch.cat([self.encoder_prompt(prompt, 0).hidden_states[-2],
                                        self.encoder_prompt(prompt, 1).hidden_states[-2]], dim=-1)
            pooled_prompt_embeds = self.encoder_prompt(prompt, 1)[0]
        else:
            text_embeddings = self.encoder_prompt(prompt, 0)[0]
            pooled_prompt_embeds =  text_embeddings

        
        return text_embeddings, pooled_prompt_embeds

    def decode_latents(self, latents):
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def compute_downsampling_size(self, image, scale_factor):
        B, C, H, W = image.shape
        
        # Calculating new dimensions based on scale_factor
        new_H = math.floor(H * scale_factor)
        new_W = math.floor(W * scale_factor)
        return (new_H, new_W)

    def string_to_number(self, s, num_bytes=4):
        hash_object = hashlib.md5(s.encode())
        hex_dig = hash_object.hexdigest()[:num_bytes * 2]
        return int(hex_dig, 16)


    def make_denoised_background(self, size, t, id=0, white=False):
        with torch.autocast('cuda', enabled=False): # vae encoder is sensetive to precision
            H, W = size

            id = f"{id}_{H}_{W}_{t}"
            if H == 0 or W == 0:
                return torch.zeros(1, 4, H, W).to(self.device)            

            self.seed_everything(self.string_to_number(id), seed_np=False) # make sure same background and noise are sampled at each iteration 
            random_bg = torch.rand(1, 3, device=self.device)[:, :, None, None].repeat(1, 1, H * 8, W * 8)
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                random_bg = random_bg.float()
            random_bg_encoded = self.vae.encode(random_bg).latent_dist.sample() * self.vae.config.scaling_factor
            
            noise = [random_bg_encoded, torch.randn_like(random_bg_encoded)]
            timesteps = t.long()
            random_bg_encoded_t = self.scheduler.add_noise(noise[0], noise[1], timesteps)
            self.seed_everything(np.random.randint(100000), seed_np=False)

            return random_bg_encoded_t
        
    def background_pad(self, input_tensor, pad_sequence, t, white=False):
        # Ensure pad_sequence length is even and divides evenly by 2 (for pairs)
        assert len(pad_sequence) % 2 == 0, "pad_sequence length must be even."
        output_tensor = input_tensor
        B, C, H, W = output_tensor.shape
        
        for dim, (pad_before, pad_after) in enumerate(zip(pad_sequence[0::2], pad_sequence[1::2])):
            dim = len(input_tensor.shape) - dim - 1
            pad_shape_before = list(output_tensor.shape)
            pad_shape_after = list(output_tensor.shape)
            pad_shape_before[dim] = pad_before
            pad_shape_after[dim] = pad_after
            
            pad_tensor_before = self.make_denoised_background(size=(pad_shape_before[-2], pad_shape_before[-1]),
                                                               t=t,
                                                               id=f"{dim}_1",
                                                            white=white).repeat(B, 1, 1, 1).to(input_tensor)
            
            pad_tensor_after =  self.make_denoised_background(size=(pad_shape_after[-2], pad_shape_after[-1]),
                                                               t=t,
                                                               id=f"{dim}_2",
                                                               white=white).repeat(B, 1, 1, 1).to(input_tensor)
            
            output_tensor = torch.cat([pad_tensor_before, output_tensor, pad_tensor_after], dim=dim)
            
        return output_tensor

    def unet_step(self, latent, t, text_embeds, 
                  add_text_embeds,
                  crops_coords_top_left=(0, 0)):
        B, C, H, W = latent.shape

        d_H, d_W = 64, 64 
        if self.sd_version.startswith('XL'):
            d_H, d_W = 128, 128

        latent = self.scheduler.scale_model_input(latent, t)

        # adjust latent size with padding
        h_p, w_p = max(d_H - latent.shape[-2], 0), max(d_W - latent.shape[-1], 0)
        l_p, r_p, t_p, b_p = w_p//2, w_p - w_p//2, h_p//2, h_p-h_p//2
        if h_p > 0 or w_p > 0:
            padded_latent = self.background_pad(latent, (l_p, r_p, t_p, b_p), t, white=False)

        else:
            padded_latent = latent

        if self.sd_version.startswith('XL'):
            original_size = target_size = self.default_size


            add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=text_embeds.dtype).to(text_embeds.device)
            add_time_ids = add_time_ids.to(self.device).repeat(padded_latent.shape[0], 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
            nxt_latent = self.unet(padded_latent, t, encoder_hidden_states=text_embeds, added_cond_kwargs=added_cond_kwargs)['sample']

        
        else:
            nxt_latent = self.unet(padded_latent, t, encoder_hidden_states=text_embeds)['sample']

        # crop latent 
        if h_p > 0 or w_p > 0: 
            nxt_latent = nxt_latent[:, :, t_p:nxt_latent.shape[-2] - b_p, l_p:nxt_latent.shape[-1] - r_p]

        return nxt_latent

    @timelog.time_function
    def obtain_latent_direction(self, latent, t, text_embeds, add_text_embeds, return_scores=False):
        downsampled_latent_model_input = torch.cat([latent] * 2)
        downsampled_noise = self.unet_step(downsampled_latent_model_input, t,
                                           text_embeds=text_embeds, add_text_embeds=add_text_embeds)
        downsampled_noise_pred_uncond, downsampled_noise_pred_cond = downsampled_noise.chunk(2)
        direction = (downsampled_noise_pred_cond - downsampled_noise_pred_uncond)
        if return_scores:
            return direction, {"uncond_score":downsampled_noise_pred_uncond, "cond_score":downsampled_noise_pred_cond}
        return direction


    def restore_mask_shape(self, M, A, dim):
        i, j = 0, 0
        R = []
        while i < M.shape[dim]:
            if j < len(A) and i == A[j]:
                if dim == 0:
                    R.append(M[i:i+1, :])
                    R.append(M[i+1:i+2, :])
                else:
                    R.append(M[:, i:i+1])
                    R.append(M[:, i+1:i+2])
                j += 2
            else:
                if dim == 0:
                    R.append(M[i:i+1, :] | M[i+1:i+2, :])
                else:
                    R.append(M[:, i:i+1] | M[:, i+1:i+2])
            i += 2
                
        return torch.cat(R, dim=dim)


    def to_even_rational(self, f, max_block_sz=32):
        frac = Fraction(f).limit_denominator(max_block_sz)
        if frac.numerator % 2 != 0 or frac.denominator % 2 != 0:
            frac = Fraction(f).limit_denominator(max_block_sz//2)
            
        if frac.numerator % 2 != 0 or frac.denominator % 2 != 0:
            return frac.numerator * 2, frac.denominator * 2

        return frac.numerator, frac.denominator

    def get_keep_blocks(self, tensor, n):
        num_blocks = n // 2

        mask = torch.ones_like(tensor, dtype=torch.bool)

        interval = len(tensor) // (num_blocks + 1)
        # interval should be even
        if interval % 2 != 0:
            interval += 1


        cnt_blocks = 0
        masked_blocks = []
        for i in range(num_blocks):
            start_index = (i + 1) * interval - 1
            masked_blocks.extend([start_index - 1 - cnt_blocks * 2, start_index + 2 - (cnt_blocks+1) * 2])
            mask[start_index:start_index + 2] = False
            cnt_blocks += 1

        result = tensor[mask]

        return result, torch.tensor(masked_blocks).to(result.device)
    
    @timelog.time_function
    def random_sample_exclude_mask(self, N, mask=None, hi=4, max_iteration=50):
        random_indices = torch.randint(0, hi, (N,))

        if mask is not None:
            invalid = mask[torch.arange(N), random_indices]
            M = invalid.sum() 
            while M > 0 and max_iteration > 0:
                random_indices[invalid] = torch.randint(0, hi, (M,))
                invalid = mask[torch.arange(N), random_indices]
                M = invalid.sum()
                max_iteration -= 1
    
            # For any remaining zeros (if all 1-4 were excluded), just randomize between 1 and 4. This risks repeated elements
            invalid = mask[torch.arange(N), random_indices]
            M = invalid.sum()
            if M > 0:
                random_indices[invalid] = torch.randint(0, hi, (M,))
    
        return random_indices
    
    @timelog.time_function
    def random_downsample(self, input, downsample_factor, exclude_mask=None, prev_random_indices=None, drop_p=0.8, nearest=False):
        # Input: Batch x Channels x Height x Width tensor
        random_indices = None
        B, C, H, W = input.shape
        new_H, new_W = H // downsample_factor, W // downsample_factor

        mask = torch.zeros((H, W), dtype=torch.bool, device=input.device)
        
        ret = []
        for c in range(input.shape[1]):
            unfold = F.unfold(input[:, c:c+1, :, :], kernel_size=downsample_factor, stride=downsample_factor) 
            if random_indices is None:
                if nearest:
                    random_indices = torch.zeros(unfold.size(2), device=input.device, dtype=torch.long)
                else:    
                    random_indices = self.random_sample_exclude_mask(N=unfold.size(2), mask=exclude_mask, hi=downsample_factor ** 2).to(input.device)
                
                if prev_random_indices is not None:
                    drop_mask = torch.randint(0, 101, (unfold.size(2),), device=input.device)
                    drop_mask[drop_mask <= (100 * drop_p)] = 0
                    drop_mask[drop_mask >= (100 * drop_p)] = 1
                    random_indices = random_indices * drop_mask + prev_random_indices * (1 - drop_mask)

            downsampled = unfold[:, random_indices, torch.arange(unfold.size(2))]
            output_shape = (input.size(0), 1, input.size(2) // downsample_factor, input.size(3) // downsample_factor)
            ret.append(downsampled.view(output_shape))

        idx_h, idx_w = torch.meshgrid(torch.arange(new_H, device=input.device), torch.arange(new_W, device=input.device), indexing='ij')
        idx_h, idx_w = idx_h.contiguous(), idx_w.contiguous()

        sampled_h = (idx_h * downsample_factor + random_indices.reshape(idx_h.shape[0], idx_h.shape[1]) // downsample_factor).view(-1)
        sampled_w = (idx_w * downsample_factor + random_indices.reshape(idx_h.shape[0], idx_h.shape[1]) % downsample_factor).view(-1)

        mask[sampled_h, sampled_w] = True
        
        return torch.cat(ret, dim=1), mask, random_indices
    
    @timelog.time_function
    def random_nearest_downsample(self, input, downsample_size, prev_random_indices=None, exclude_mask=None, drop_p=0.8, nearest=False):
        # Future TODO: enable this function for downsample_factor > 2

        # scale input to 2x
        resized = self.nearest_interpolate(input, size=(input.shape[2] * 2, input.shape[3] * 2), mode='nearest')

        # scale result to downsample_size * 2
        r_n_keep, r_block_sz = self.to_even_rational(downsample_size[0] / input.shape[2])
        r_n_remove = r_block_sz-r_n_keep # rows to remove per block to reach downsample_factor * 2 
        c_n_keep, c_block_sz = self.to_even_rational(downsample_size[1] / input.shape[3])
        c_n_remove = c_block_sz-c_n_keep # cols to remove per block to reach downsample_factor * 2 

        r_num_blocks = ((downsample_size[0] * 2) // r_n_keep)
        c_num_blocks = ((downsample_size[1] * 2) // c_n_keep) 
        if r_num_blocks *  r_block_sz > input.shape[2] * 2:
            r_num_blocks -= 1
        if c_num_blocks *  c_block_sz > input.shape[3] * 2:
            c_num_blocks -= 1
            
        r_blocks = r_num_blocks * r_block_sz # number of row blocks in 2x input
        c_blocks = c_num_blocks * c_block_sz # number of column blocks in 2x input

        
        if 'row_indices' not in self.random_downasmple_pre:
            row_indices = torch.arange(0, r_blocks, r_block_sz)
            offsets, r_masked_blocks = self.get_keep_blocks(torch.arange(r_block_sz), r_n_remove) # indices to keep and remove in each block
            row_indices = (row_indices.view(-1, 1) + offsets).view(-1)
            row_indices = row_indices[row_indices < input.shape[2] * 2]
            self.random_downasmple_pre['row_indices'] = row_indices

            mask_row_indices = torch.arange(0, downsample_size[0]*2, r_n_keep)
            mask_row_indices = (mask_row_indices.view(-1, 1) + r_masked_blocks).view(-1)
            self.random_downasmple_pre['mask_row_indices'] = mask_row_indices
            
        if 'col_indices' not in self.random_downasmple_pre:
            col_indices = torch.arange(0, c_blocks, c_block_sz)
            offsets, c_masked_blocks = self.get_keep_blocks(torch.arange(c_block_sz), c_n_remove)
            col_indices = (col_indices.view(-1, 1) + offsets).view(-1)
            col_indices = col_indices[col_indices < input.shape[3] * 2]
            self.random_downasmple_pre['col_indices'] = col_indices

            mask_col_indices = torch.arange(0, downsample_size[1]*2, c_n_keep)
            mask_col_indices = (mask_col_indices.view(-1, 1) + c_masked_blocks).view(-1)
            self.random_downasmple_pre['mask_col_indices'] = mask_col_indices

        
        
        row_indices = self.random_downasmple_pre['row_indices']
        col_indices = self.random_downasmple_pre['col_indices']
        r_remain = downsample_size[0]*2 - len(row_indices)
        c_remain = downsample_size[1]*2 - len(col_indices)
        rows = torch.cat([resized[:, :, row_indices, :], resized[:, :, r_blocks:r_blocks+r_remain]], dim=2)
        resized = torch.cat([rows[:, :, :, col_indices], rows[:, :, :, c_blocks:c_blocks+c_remain]], dim=3)


        downsampled, mask, prev_random_indices = self.random_downsample(resized,
                                                                        downsample_factor=2, 
                                                                        drop_p=drop_p,
                                                                        prev_random_indices=prev_random_indices,
                                                                        exclude_mask=exclude_mask,
                                                                        nearest=nearest) # Using the previous random_downsample function
        mask_rows = self.restore_mask_shape(mask, self.random_downasmple_pre['mask_row_indices'], 0)
        mask = self.restore_mask_shape(mask_rows, self.random_downasmple_pre['mask_col_indices'], 1)
        
        if input.shape[2] > mask.shape[0]:
            mask = torch.cat([mask, torch.zeros(input.shape[2] - mask.shape[0], mask.shape[1]).to(torch.bool).to(mask.device)], dim=0)
        if input.shape[3] > mask.shape[1]:
            mask = torch.cat([mask, torch.zeros(mask.shape[0], input.shape[3] - mask.shape[1]).to(torch.bool).to(mask.device)], dim=1)

        return downsampled, mask, prev_random_indices
    

    @timelog.time_function
    def fill_in_from_downsampled_direction(self, target_direction, downsampled_direction, mask, fill_all=False):
        B, C, H, W = target_direction.shape
        upsampled_direction = self.nearest_interpolate(downsampled_direction, size=(target_direction.size(2), target_direction.size(3)))
        target_direction = torch.where(mask, upsampled_direction, target_direction)
        
        if fill_all:
            if self.verbose:
                print(f"[INFO] {(torch.sum(~torch.isnan(target_direction)) / target_direction.numel()) * 100:.2f}% of the target direction was filled with resampling")
            
            nan_mask = torch.isnan(target_direction)
            target_direction = torch.where(nan_mask, upsampled_direction, target_direction)


        return target_direction

    @timelog.time_function
    def approximate_latent_direction_w_resampling(self, latent, t, text_embeds, add_text_embeds,
                                                factor=None, downsample_size=None, resampling_steps=6,
                                                drop_p=0.7, fix_initial=True):
        
        exclude_mask = None
        target_direction = torch.full_like(latent, float('nan')).half()

        approximation_info = {}
        approximation_info['init_downsampled_latent'] = None
        prev_random_indices = None

        for step in range(resampling_steps+1):
            if downsample_size is None:
                downsample_size = self.compute_downsampling_size(latent, scale_factor=1/factor)
            
            downsampled_latent, mask, prev_random_indices = self.random_nearest_downsample(latent, downsample_size,
                                                                    prev_random_indices=prev_random_indices,
                                                                    drop_p=drop_p,
                                                                    exclude_mask=exclude_mask,
                                                                    nearest=(step==0) and fix_initial)
            


            if exclude_mask is None:
                exclude_mask = torch.zeros((len(prev_random_indices), 4), dtype=torch.bool, device=mask.device)
            exclude_mask[torch.arange(len(prev_random_indices)), prev_random_indices] = True

            if approximation_info['init_downsampled_latent'] is None:
                approximation_info['init_downsampled_latent'] = downsampled_latent.clone()
            
            direction, scores = self.obtain_latent_direction(downsampled_latent, t, text_embeds, add_text_embeds, return_scores=True)
            target_direction = self.fill_in_from_downsampled_direction(target_direction, direction, mask, fill_all=(step==resampling_steps))
            if self.verbose:
                print(f"[INFO] {(torch.sum(~torch.isnan(target_direction)) / target_direction.numel()) * 100:.2f}% of the target direction was filled after resampling step {step}")
            
        
        approximation_info['downsampled_latent'] = downsampled_latent
        approximation_info['scores'] = scores
        approximation_info['downsampled_direction'] = self.nearest_interpolate(target_direction, size=downsample_size, mode='nearest')
        
        return target_direction, approximation_info
    
    def undo_step(self, sample, timestep, generator=None):
        n = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        for i in range(n):
            if i >= self.scheduler.config.num_train_timesteps:
                continue
            t = timestep + i
            beta = self.scheduler.betas[t]

            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample
    
    def crop_with_context(self, X, a, b, c, d, S, n):
        """
        X: torch.Tensor - input image of shape (B, C, H, W)
        a, b: int - vertical cropping indices
        c, d: int - horizontal cropping indices
        S: int - stride
        n: int - number of context pixels
        """
        B, C, H, W = X.shape
        
        n_t = n_b = n_r = n_l = n 
        
        if a - n_t * S < 0:
            top_rows = np.arange(max(0, a - n_t * S), a - S + 1, S)
            n_t = len(top_rows)
            n_b = 2 * n - n_t
            bottom_rows = np.arange(b - 1 + S, min(H, b + n_b * S), S)
            n_b = len(bottom_rows)
        else:
            bottom_rows = np.arange(b - 1 + S, min(H, b + n_b * S), S)
            n_b = len(bottom_rows)
            n_t = 2 * n - n_b
            top_rows = np.arange(max(0, a - n_t * S), a - S + 1, S)
            n_t = len(top_rows)
            
        # Get the top context rows
        if c - n_l * S < 0: 
            left_cols = np.arange(max(0, c - n_l * S), c - S + 1, S)
            n_l = len(left_cols)
            n_r = 2 * n - n_l
            right_cols = np.arange(d - 1 + S, min(W, d + n_r * S), S)
            n_r = len(right_cols)
        
        else:
            right_cols = np.arange(d - 1 + S, min(W, d + n_r * S), S)
            n_r = len(right_cols)
            n_l = 2 * n - n_r
            left_cols = np.arange(max(0, c - n_l * S), c - S + 1, S)
            n_l = len(left_cols)
        
        x_inds = np.concatenate([top_rows, np.arange(a, b), bottom_rows])
        
        top_samples = X[:, :, top_rows, c:d]
        bottom_samples = X[:, :, bottom_rows, c:d]
        left_samples = X[:, :, x_inds, :][:, :, :, left_cols]
        right_samples = X[:, :, x_inds, :][:, :, :, right_cols]
        
        # Combine the contexts with the center crop
        vertical_combined = torch.cat([top_samples, X[:, :, a:b, c:d], bottom_samples], dim=2)
        final_crop = torch.cat([left_samples, vertical_combined, right_samples], dim=3)
        
        return final_crop, (n_t, n_b, n_l, n_r)


    @torch.no_grad()
    def generate(self, latent, text_embeds, add_text_embeds, guidance_scale=7.5):
        intermediate_steps_x0 = []
        with torch.autocast('cuda', enabled=(self.device.type=='cuda')):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                global_latent_model_input = torch.cat([latent] * 2)
                global_noise = self.unet_step(global_latent_model_input, t,
                                             text_embeds=text_embeds, add_text_embeds=add_text_embeds) 
                global_noise_pred_uncond, global_noise_pred_cond = global_noise.chunk(2)
                global_direction = (global_noise_pred_cond - global_noise_pred_uncond)

                global_noise_pred = global_noise_pred_uncond + guidance_scale * global_direction

                ddim_out = self.scheduler.step(global_noise_pred, t, latent)
                
                latent = ddim_out['prev_sample']
                if i % self.log_freq == 0:
                    intermediate_steps_x0.append(ddim_out['pred_original_sample'].cpu())

        return T.ToPILImage()(self.decode_latents(latent).cpu()[0]), {"inter_x0":intermediate_steps_x0}
    

    ## Copied from https://github.com/huggingface/diffusers/blob/cf03f5b7188c603ff037d686f7256d0571fbd651/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L66
    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg
    
    @timelog.time_function
    def compute_local_uncond_signal(self, latent, t, 
                                    uncond_text_embeds, negative_pooled_prompt_embeds,
                                    view_config):
        height, width = latent.shape[-2] * 8, latent.shape[-1] * 8
    
        # edge case where context pixel are not required in one dimension
        h_ws = w_ws = view_config['window_size']
        if h_ws + view_config['context_size'] >= latent.shape[2]:
            h_ws = latent.shape[2]
        
        if w_ws + view_config['context_size'] >= latent.shape[3]:
            w_ws = latent.shape[3]

        views = self.get_views(height, width, h_ws=h_ws, w_ws=w_ws, **view_config)
        local_uncond_noise_val = torch.zeros_like(latent)

        for batch_start_idx in range(0, len(views), self.view_batch_size):
            views_batch = views[batch_start_idx:batch_start_idx+self.view_batch_size]
            latent_views = []
            views_batch_wc = []
            for view in views_batch:
                h_start, h_end, w_start, w_end = view

                latent_view, (n_t, n_b, n_l, n_r) = \
                    self.crop_with_context(latent, h_start, h_end, w_start, w_end, S=1, n=view_config['context_size'] // 2)

                latent_views.append(latent_view)
                views_batch_wc.append((n_t, n_b, n_l, n_r))


            # predict the noise residual
            latent_model_input = torch.cat(latent_views)
            text_embeds_input = torch.cat([uncond_text_embeds] * len(views_batch))
            add_text_embeds_input = torch.cat([negative_pooled_prompt_embeds] * len(views_batch))
            noise_pred_uncond = self.unet_step(latent_model_input, t,
                                               text_embeds=text_embeds_input,
                                                add_text_embeds=add_text_embeds_input) 
            
            for view, view_wc, view_pred_noise in zip(views_batch, views_batch_wc, noise_pred_uncond.chunk(len(views_batch))):
                h_start, h_end, w_start, w_end = view
                n_t, n_b, n_l, n_r = view_wc
                s_h = (n_t, view_pred_noise.shape[-2] - n_b)
                s_w = (n_l, view_pred_noise.shape[-1] - n_r)

                
                non_zero_maks = local_uncond_noise_val[:, :, h_start:h_end, w_start:w_end] != 0
                local_uncond_noise_val[:, :, h_start:h_end, w_start:w_end][~non_zero_maks] = \
                            view_pred_noise[:, :, s_h[0]:s_h[1], s_w[0]:s_w[1]][~non_zero_maks].to(local_uncond_noise_val.dtype)
            

        return local_uncond_noise_val 



    @timelog.time_function
    def nearest_interpolate(self, x, size, bottom=False, right=False, mode='nearest'):
        """nearest interpolate with different corresponding pixels to choose top-left, top-right, bottom-left, or bottom-right"""
        if bottom:
            x = torch.flip(x, [2])
        if right:
            x = torch.flip(x, [3])
        
        x = F.interpolate(x, size=size, mode=mode)

        if bottom:
            x = torch.flip(x, [2])
        if right:
            x = torch.flip(x, [3])

        return x

    @timelog.time_function
    def reduced_resolution_guidance(self, global_latent, t, global_direction,
                          latent_x0_original, uncond_text_embeds, negative_pooled_prompt_embeds,
                          view_config, guidance_scale, rrg_scale,
                          factor=None, downsample_size=None, bottom=False, right=False, text_embeds=None, min_H=-0, min_W=0,
                          donwsampled_scores=None):
        if downsample_size is None:
                downsample_size = self.compute_downsampling_size(global_latent, scale_factor=1/factor)

        
        if donwsampled_scores is None:
            H, W = downsample_size

            H = max(H, min_H)
            W = max(W, min_W)
            
            global_latent_downsampled = self.nearest_interpolate(global_latent, size=(H, W), bottom=bottom, right=right)
            input_latent = global_latent_downsampled
            direction = self.nearest_interpolate(global_direction, size=(H, W), bottom=bottom, right=right)
            local_uncond_noise = self.compute_local_uncond_signal(input_latent, t, 
                                    uncond_text_embeds, negative_pooled_prompt_embeds,
                                    view_config)

            
        else:
            input_latent = donwsampled_scores['latent']
            direction = donwsampled_scores['direction']
            local_uncond_noise = donwsampled_scores['uncond_score']
            H, W = direction.shape[-1], direction.shape[-2]

            H = max(H, min_H)
            W = max(W, min_W)

        global_noise_pred = local_uncond_noise + guidance_scale * direction

        ddim_out = self.scheduler.step(global_noise_pred, t, input_latent)
        ref_x0_original = ddim_out['pred_original_sample']
        ref_x0_original_upsampled = self.nearest_interpolate(ref_x0_original, 
                                                  size=(latent_x0_original.shape[-2], latent_x0_original.shape[-1]), 
                                                  mode='nearest')
        
        added_grad_list = []
        for j in range(len(global_latent)):
            with torch.enable_grad():
                dummy_pred = latent_x0_original[j:j+1].clone().detach()
                dummy_pred = dummy_pred.requires_grad_(requires_grad=True)
                
                loss = rrg_scale * torch.nn.functional.mse_loss(ref_x0_original_upsampled[j:j+1], dummy_pred)
                
                loss.backward()
                added_grad = dummy_pred.grad.clone() * -1.
                added_grad_list.append(added_grad)
        
        added_grad = torch.cat(added_grad_list)
        
        return added_grad, {"x0" : [ref_x0_original], "rrg_latent_out": [ddim_out['prev_sample']]}
        

    def get_downsample_size(self, H, W):
        if 'XL' in self.sd_version:
            factor = max(H, W) / 1024
        else:
            factor = max(H, W) / 512
        
        factor = max(factor, 1)
        return (int((H // factor) // 8), int((W // factor) // 8))
    
    @torch.no_grad()
    def generate_image(self, prompts, negative_prompts='', 
                       height=768, width=768, 
                       num_inference_steps=50,
                       guidance_scale=10.0, 
                       resampling_steps=20,
                       new_p=0.3, rrg_stop_t=0.2, 
                       rrg_init_weight=1000,
                       rrg_scherduler_cls=CosineScheduler,
                       cosine_scale=3.0,
                       repaint_sampling=True,
                       progress=tqdm,
                       grid=False):
        
        self.random_downasmple_pre = {}
        downsample_size = self.get_downsample_size(height, width)
        self.default_size = (4*height, 4*width) 
        view_config = self.view_config

        if rrg_scherduler_cls == CosineScheduler:
            rrg_scheduler = rrg_scherduler_cls(steps=num_inference_steps - int(num_inference_steps * rrg_stop_t),
                                    cosine_scale=cosine_scale,
                                    factor=rrg_init_weight)
        else:
            rrg_scheduler = rrg_scherduler_cls(steps=num_inference_steps - int(num_inference_steps * rrg_stop_t),
                                start_val=rrg_init_weight,
                                stop_val=0)
        
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        uncond_text_embeds, negative_pooled_prompt_embeds = self.get_text_embeds(negative_prompts)
        cond_text_embeds, pooled_prompt_embeds= self.get_text_embeds(prompts)

        text_embeds = torch.cat([uncond_text_embeds, cond_text_embeds])  # [2, 77, 768]
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        global_latent = torch.randn((len(prompts), self.unet.config.in_channels, height // 8, width // 8),
                                     device=self.device,
                                     dtype=self.torch_dtype) # we divide by 8 to get the latent dimension of stabel diffusion
        self.scheduler.set_timesteps(num_inference_steps)
        
        init_downsampled_latent = None
        intermediate_x0_imgs = []
        intermediate_cascade_x0_imgs_lst = {}

        with torch.autocast('cuda', enabled=(self.device.type=='cuda')):
            for i, t in enumerate(progress(self.scheduler.timesteps)):
                #################### Estimate directions ####################
                cur_resampling_steps = resampling_steps
                global_direction, approximation_info = self.approximate_latent_direction_w_resampling(global_latent, t, text_embeds,
                                                                                                    resampling_steps=cur_resampling_steps,
                                                                                                    downsample_size=downsample_size,
                                                                                                    add_text_embeds=add_text_embeds,
                                                                                                    drop_p=1-new_p)
                
                
                if init_downsampled_latent is None:
                    init_downsampled_latent = approximation_info['init_downsampled_latent']
            

                local_uncond_noise = self.compute_local_uncond_signal(global_latent, t, 
                                    uncond_text_embeds, negative_pooled_prompt_embeds,
                                    view_config)
                
                global_noise_pred = local_uncond_noise + guidance_scale * global_direction

                ddim_out = self.scheduler.step(global_noise_pred, t, global_latent)
                latent_x0_original = ddim_out['pred_original_sample']
                global_latent_nxt = ddim_out['prev_sample']
                rrg_cfg = guidance_scale

                if repaint_sampling and cur_resampling_steps > 0 and i < len(self.scheduler.timesteps) - 1:
                    global_latent = ddim_out['prev_sample']
                    global_latent = self.undo_step(global_latent, self.scheduler.timesteps[i+1])
                    rrg_cfg = guidance_scale / 3

                    global_direction, approximation_info = self.approximate_latent_direction_w_resampling(global_latent, t, text_embeds,
                                                                                                            resampling_steps=0,
                                                                                                            downsample_size=downsample_size,
                                                                                                            add_text_embeds=add_text_embeds,
                                                                                                            drop_p=1-new_p)
                    
                    local_uncond_noise = self.compute_local_uncond_signal(global_latent, t, 
                                    uncond_text_embeds, negative_pooled_prompt_embeds,
                                    view_config)

                    global_noise_pred = local_uncond_noise + rrg_cfg * global_direction
                    ddim_out = self.scheduler.step(global_noise_pred, t, global_latent)
                    latent_x0_original = ddim_out['pred_original_sample']
                    global_latent_nxt = ddim_out['prev_sample']

                if self.verbose and i % self.log_freq == 0:
                    intermediate_x0_imgs.append(latent_x0_original.cpu())

                cascade_dir = torch.zeros_like(global_latent_nxt)
                if rrg_scheduler(i) > 10:
                    donwsampled_scores = {"latent":approximation_info['downsampled_latent'], 
                                  "uncond_score": approximation_info['scores']['uncond_score'],
                                  "direction": approximation_info['downsampled_direction']}

                    cascade_dir, cascade_info = self.reduced_resolution_guidance(global_latent, t, global_direction,
                                                        latent_x0_original, uncond_text_embeds, negative_pooled_prompt_embeds,
                                                        view_config, downsample_size=downsample_size, rrg_scale=rrg_scheduler(i),
                                                        guidance_scale=rrg_cfg, text_embeds=text_embeds, 
                                                        donwsampled_scores=donwsampled_scores, bottom=False, right=False)

                    if self.verbose and i % self.log_freq == 0:
                            lst = intermediate_cascade_x0_imgs_lst.get('rrg', [])
                            lst.append(cascade_info['x0'][0].cpu())
                            intermediate_cascade_x0_imgs_lst['rrg'] = lst

                global_latent = global_latent_nxt + cascade_dir
        
        #upcast vae 
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()
        
        decode_bs = 1
        image_log = {}
        if self.verbose:
            if init_downsampled_latent is not None:
                image_log['global_img'], generation_info  = self.generate(init_downsampled_latent, text_embeds, add_text_embeds, guidance_scale=guidance_scale)
                if 'inter_x0' in generation_info:
                    inter_x0_decoded = torch.cat([self.decode_latents(torch.cat(generation_info['inter_x0'][i:i+decode_bs]).to(self.device)) \
                                            for i in range(0, len(generation_info['inter_x0']), decode_bs)])
                    image_log['global_img_inter_x0_imgs'] = T.ToPILImage()(make_grid(inter_x0_decoded,
                                                                        nrows=len(generation_info['inter_x0']),
                                                                        normalize=False).cpu())
            

            if intermediate_x0_imgs:
                inter_x0_decoded = torch.cat([self.decode_latents(torch.cat(intermediate_x0_imgs[i:i+decode_bs]).to(self.device)) \
                                            for i in range(0, len(intermediate_x0_imgs), decode_bs)])
                inter_x0_decoded = torch.clip(inter_x0_decoded, 0, 1)
                image_log['intermediate_x0_imgs'] = T.ToPILImage()(make_grid(inter_x0_decoded,
                                                                        nrows=len(intermediate_x0_imgs),
                                                                        normalize=False).cpu())

            image_log['intermediate_cascade_x0_imgs'] = {}
            for factor, intermediate_cascade_x0_imgs in intermediate_cascade_x0_imgs_lst.items():
                inter_cascade_x0_decoded = torch.cat([self.decode_latents(torch.cat(intermediate_cascade_x0_imgs[i:i+decode_bs]).to(self.device)) \
                                            for i in range(0, len(intermediate_cascade_x0_imgs), decode_bs)])
                image_log['intermediate_cascade_x0_imgs'][factor] = T.ToPILImage()(make_grid(inter_cascade_x0_decoded,
                                                                        nrows=len(intermediate_cascade_x0_imgs),
                                                                        normalize=False).cpu())
        
        # Img latents -> imgs
        imgs = torch.cat([self.decode_latents(global_latent[i:i+decode_bs]) for i in range(0, len(global_latent), decode_bs)])

        if grid:
            imgs = [make_grid(imgs, nrows=len(imgs), normalize=False)]
        imgs = [T.ToPILImage()(img.cpu()) for img in imgs] 

        return imgs, image_log



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Envision a portrait of a horse, framed by a blue headscarf with muted tones of rust and cream. she has brown-colored eyes. Her attire, simple yet dignified")
    parser.add_argument('--negative', type=str, default='blurry, ugly, duplicate, no details, deformed')
    parser.add_argument('--sd_version', type=str, default='XL1.0', choices=['1.4', '1.5', '2.0', '2.1', 'XL1.0'],
                        help="stable diffusion version stable diffusion version ['1.4', '1.5', '2.0', '2.1', or 'XL1.0'] or a model key for a huggingface stable diffusion version")
    parser.add_argument('--H', type=int, default=1920)
    parser.add_argument('--W', type=int, default=1080)
    parser.add_argument('--low_memory', type=bool, default=False, help="run with half percision on low memeory mode")
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_sampled', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--cosine_scale', type=float, default=10.0, help='effective only with CosineScheduler')
    parser.add_argument('--rrg_scale', type=float, default=1000)
    parser.add_argument('--resampling_steps', type=int, default=7)
    parser.add_argument('--new_p', type=float, default=0.3)
    parser.add_argument('--rrg_stop_t', type=float, default=0.2)
    parser.add_argument('--view_batch_size', type=int, default=16)
    parser.add_argument('--outdir', type=str, default='../release_results_log/')
    parser.add_argument('--make_grid', type=bool, default=False, help="make a grid of the output images")
    parser.add_argument('--repaint_sampling', type=bool, default=True, help="")
    parser.add_argument('--exp', type=str, default='ElasticDiffusion', help='experiment tag')
    parser.add_argument('--tag', type=str, default='', help='identifier experiment tag')
    parser.add_argument('--log_freq', type=int, default=5, help="log frequency of intermediate diffusion steps")
    parser.add_argument('--verbose', type=bool, default=True)
    opt = parser.parse_args()
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if opt.verbose:
        timelog.sync_gpu = opt.verbose # get accurate time log

    sd = ElasticDiffusion(device,
                          opt.sd_version,
                          verbose=opt.verbose,
                          log_freq=opt.log_freq,
                          view_batch_size=opt.view_batch_size,
                          torch_dtype=torch.float16 if opt.low_memory else torch.float32) 

    sd.seed_everything(opt.seed)

    prompts = [opt.prompt] * opt.num_sampled
    imgs, image_log = sd.generate_image(prompts=prompts, negative_prompts=opt.negative,
                            height=opt.H, width=opt.W, 
                            num_inference_steps=opt.steps, 
                            grid=opt.make_grid,
                            guidance_scale=opt.guidance_scale, 
                            resampling_steps=opt.resampling_steps,
                            new_p=opt.new_p,
                            cosine_scale = opt.cosine_scale,
                            rrg_init_weight = opt.rrg_scale,
                            rrg_stop_t = opt.rrg_stop_t,
                            repaint_sampling=opt.repaint_sampling)

    if opt.verbose:
        timelog.print_results()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_dir = os.path.join(opt.outdir, opt.exp, f"{current_time}_{str(opt.seed)}")
    os.makedirs(save_dir, exist_ok=True)
    # save image
    for i, img in enumerate(imgs):
        img.save(f"{save_dir}/{i}.png")
    
    for key, imgs in image_log.items():
        if isinstance(imgs, dict):
            [img.save(f"{save_dir}/{key}_{label}.png") for label, img in image_log[key].items()]
        else:
            image_log[key].save(f"{save_dir}/{key}.png")
    
    # save meta
    with open(f"{save_dir}/args.txt", 'w') as f:
        args_str = '\n'.join(['{}: {}'.format(k, v) for k, v in vars(opt).items()])
        f.write(args_str)

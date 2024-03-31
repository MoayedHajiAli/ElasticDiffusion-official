from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid 
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class VanillaLDM(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
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
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.float32).to(self.device)
        self.tokenizer = [CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")]
        self.text_encoder = [CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)]
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        if self.sd_version == 'XL1.0':
            self.text_encoder.append(CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2").to(self.device))
            self.tokenizer.append(CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2"))

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion form {model_key}!')
    
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
    
    
    def unet_step(self, latent, t, text_embeds, 
                  add_text_embeds,
                  crops_coords_top_left=(0, 0),
                  default_size = (512, 512)):
        
        latent = self.scheduler.scale_model_input(latent, t)

        if self.sd_version.startswith('XL'):
            original_size = target_size = default_size
            add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=text_embeds.dtype).to(text_embeds.device)
            add_time_ids = add_time_ids.to(device).repeat(latent.shape[0], 1)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            return self.unet(latent, t, encoder_hidden_states=text_embeds, added_cond_kwargs=added_cond_kwargs)['sample']

        else:
            return self.unet(latent, t, encoder_hidden_states=text_embeds)['sample']
        
    def encoder_prompt(self, prompt, encoder_id):
        text_input = self.tokenizer[encoder_id](prompt, padding='max_length', max_length=self.tokenizer[encoder_id].model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder[encoder_id](text_input.input_ids.to(self.device), output_hidden_states=True)
        return text_embeddings

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        if self.sd_version == 'XL1.0':
            text_embeddings = torch.cat([self.encoder_prompt(prompt, 0).hidden_states[-2],
                                        self.encoder_prompt(prompt, 1).hidden_states[-2]], dim=-1)
            pooled_prompt_embeds = self.encoder_prompt(prompt, 1)[0]
        else:
            text_embeddings = self.encoder_prompt(prompt, 0)[0]
            pooled_prompt_embeds =  text_embeddings

        return text_embeddings, pooled_prompt_embeds
    

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    

    @torch.no_grad()
    def generate_image(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=10.0, num_samples=1, grid=True, downsample=1, save_dir=''):

        if isinstance(prompts, str):
            prompts = [prompts] * num_samples

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        # Prompts -> text embeds
        uncond_text_embeds, negative_pooled_prompt_embeds = self.get_text_embeds(negative_prompts)
        cond_test_embeds, pooled_prompt_embeds= self.get_text_embeds(prompts)

        text_embeds = torch.cat([uncond_text_embeds, cond_test_embeds])  # [2, 77, 768]
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)


        # Define panorama grid and get views
        global_latent = torch.randn((num_samples, self.unet.config.in_channels, height // 8, width // 8), device=self.device) # we divide by 8 to get the latent dimension of stabel diffusion
        self.scheduler.set_timesteps(num_inference_steps)


        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):

                # take step in global latent
                global_latent_model_input = torch.cat([global_latent] * 2)

                # predict the noise residual
                noise_pred = self.unet_step(global_latent_model_input, t, text_embeds=text_embeds, add_text_embeds=add_text_embeds, default_size=(height, width))

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                direction = (noise_pred_cond - noise_pred_uncond)


                # PCA visualization
                pca = PCA(n_components=3)
                normalize_direction = (direction - direction.min()) / (direction.max() - direction.min()) * 2 - 1
                class_direction_score = normalize_direction[0].permute(1, 2, 0).view(-1, 4).cpu().numpy()
                
                pca.fit(class_direction_score)
                reduced = pca.transform(class_direction_score)[None, ...]

                patch_size = 8
                patch_h_num = 64
                patch_w_num = 64
                pca_image = reduced.reshape(patch_h_num, patch_w_num, 3)
                pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
                h, w, _ = pca_image.shape
                pca_image = Image.fromarray(np.uint8(pca_image * 255))
                pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
                pca_image.save(f"{save_dir}/pca_cls_dir_{i}.png")

                # PCA visualization
                pca = PCA(n_components=3)
                unconditional_score = noise_pred_uncond[0].permute(1, 2, 0).view(-1, 4).cpu().numpy()
                pca.fit(unconditional_score)
                reduced = pca.transform(unconditional_score)[None, ...]
                patch_size = 8
                patch_h_num = 64
                patch_w_num = 64
                pca_image = reduced[:, 0:].reshape(patch_h_num, patch_w_num, 3) # why get rid of the first features? 
                pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
                h, w, _ = pca_image.shape
                pca_image = Image.fromarray(np.uint8(pca_image * 255))
                pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
                pca_image.save(f"{save_dir}/pca_uncond_{i}.png")

                global_noise_pred = noise_pred_uncond + guidance_scale * direction
                ddim_out = self.scheduler.step(global_noise_pred, t, global_latent)
                global_latent = ddim_out['prev_sample']
                noise_free_latent = ddim_out['pred_original_sample']
                noise_free_sample = self.decode_latents(noise_free_latent)[0]
                inter_sample = T.ToPILImage()(noise_free_sample)
                inter_sample.save(f"{save_dir}/inter_noise_free_sample_{i}.png")

 
        # Img latents -> imgs
        if isinstance(global_latent, list):
            imgs = [self.decode_latents(latent.unsqueeze(0)).squeeze(0) for latent in global_latent]  
        else:
            imgs = self.decode_latents(global_latent)  

        if grid:
            imgs = [make_grid(imgs, nrows=len(imgs), normalize=False)]
        imgs = [T.ToPILImage()(img.cpu()) for img in imgs]

        return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Envision a realistic portrait of a young woman with red hair and green eyes, highly detailed, high resolution')
    parser.add_argument('--negative', type=str, default='Blurry, ugly, deformed')
    parser.add_argument('--sd_version', type=str, default='1.4', choices=['1.4', '1.5', '2.0', '2.1', 'XL1.0'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='results_log/pca_visualization')
    parser.add_argument('--make_grid', type=bool, default=False, help="make a grid of the output images")
    parser.add_argument('--exp', type=str, default='vanilla_dm', help='experiment tag')
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')


    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_dir = os.path.join(opt.outdir, opt.exp, current_time, str(opt.downsample))
    os.makedirs(save_dir, exist_ok=True)

    sd = VanillaLDM(device, opt.sd_version)
    imgs = sd.generate_image(opt.prompt, opt.negative, opt.H,
                              opt.W, opt.steps, grid=opt.make_grid,
                                num_samples=opt.num_samples, downsample=opt.downsample, save_dir=save_dir)

    
    # save image
    for i, img in enumerate(imgs):
        img.save(f"{save_dir}/{i}.png")
    

    # Assuming you are using a CUDA-capable GPU
    torch.cuda.synchronize() # Wait for CUDA operations to finish

    print(f"Total memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    print(f"Memory cached: {torch.cuda.memory_cached() / 1024**2} MB")

    # Reset peak memory stats
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
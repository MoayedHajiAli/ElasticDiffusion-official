import time
import gradio as gr
from elastic_diffusion_w_controlnet import ElasticDiffusion
import torch
import argparse

loaded_sd_version = 'XL1.0'
controlnet_model = 'canny'
device = torch.device('cuda')
pipe = ElasticDiffusion(device, loaded_sd_version, controlnet_model, verbose=False)

def generate_image_fn(condition_image,
                      prompt:str, 
                      negative_prompts:str,
                      controlnet_model:str,
                      img_width:int, 
                      img_height:int=512,
                      resampling_steps:int=20,
                      resampling_keep_p:float=0.8,
                      rrg_scale:float=200,
                      guidance_scale:float=10.0,
                      controlnet_conditioning_scale:float=0.5,
                      _=None, 
                      cosine_scale:float=10.0,
                      view_batch_size:int=16,
                      num_inference_steps:int=50,
                      patch_size:int=64,
                      seed:int=0,
                      low_vram:bool=False,
                      tiled_decoder:bool=False,
                      progress=gr.Progress()) -> list:
    
    global loaded_sd_version, pipe
    
    # assert (not (sd_version != 'XL1.0' and (img_height > 1024 or img_width > 1024))),  "[ERROR] Invalid Hyper-Paramters: We currently support only up to 2X higher than the training resolution of the pre-trained diffusion model. To generate images with width > 1024 or heigh > 1024, please select XL-1.0 as the stable diffusion version."
    assert (not (img_height % 8 != 0 or img_width % 8 != 0)), "[ERROR] Invalid Hyper-Paramters: Image height and width must be divisable by 8"

    # prepare conditioning image
    ds = pipe.get_downsample_size(img_height, img_width)
    condition_image = condition_image.resize((ds[1] * pipe.vae_scale_factor, ds[0] * pipe.vae_scale_factor)).convert("RGB")
    condition_image = pipe.process_condition_image(condition_image, pipe.controlnet_model)

    start_time = time.time()
    pipe.seed_everything(int(seed))
    pipe.view_batch_size = int(view_batch_size)
    pipe.set_view_config(patch_size=patch_size)

    if pipe.controlnet_model != controlnet_model or pipe.low_vram != low_vram:
        pipe = ElasticDiffusion(device, loaded_sd_version, controlnet_model, low_vram=low_vram, verbose=False)

    images, log_info = pipe.generate_image(prompts=prompt, condition_image=condition_image, 
                            negative_prompts=negative_prompts,
                            height=img_height, width=img_width, 
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=guidance_scale, 
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            resampling_steps=resampling_steps,
                            new_p=resampling_keep_p,
                            rrg_init_weight=rrg_scale,
                            rrg_stop_t=0.4,
                            repaint_sampling=True,
                            cosine_scale=float(cosine_scale),
                            tiled_decoder=tiled_decoder,
                            progress=progress.tqdm)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
    return images



parser = argparse.ArgumentParser(description='Run Gradio app')
parser.add_argument('--port', type=int, default=7860, help='Port to run the Gradio app on')
args = parser.parse_args()

description = """ """
article = ""
gr.Interface(
    generate_image_fn,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(
            label="Prompt",
            max_lines=3,
            placeholder="a photo of the dolomites",
        ),
        gr.Textbox(
            label="Negative Prompt",
            value="blurry, ugly, duplicate, no details, deformed",
            max_lines=3,
        ),
        gr.Dropdown(
            ["canny", "depth"], label="Condition type", value='canny', info="This condition will be extracted from the input image and guide the image generation"
        ),
        gr.Slider(label="Width", value=1024, minimum=128, maximum=2048, step=128),
        gr.Slider(label="Height", value=1024, minimum=128, maximum=2048, step=128),
        gr.Slider(label="Resampling: Steps", value=10, minimum=0, maximum=39, step=3),
        gr.Slider(label="Resampling: Percentage of newly sampled pixels", value=0.3, minimum=0.1, maximum=0.5, step=0.1),
        gr.Slider(label="Reduced Resolution Guidance Scale", value=200, minimum=0, maximum=4000, step=100),
        gr.Slider(label="Classifer-Free Guidance Scale", value=10, minimum=5, maximum=12, step=1),
        gr.Slider(label="ControlNet Guidance Scale", value=0.5, minimum=0.1, maximum=2.0, step=0.1),
        gr.HTML("<p>Additional Hyper-Parameters<p>"),
        gr.Textbox(label="Cosine Scale", value=10.0, max_lines=1),
        gr.Slider(label="View Batch Size", value=16, minimum=1, maximum=64, step=1),
        gr.Slider(label="Number of Inference Steps", value=50, minimum=40, maximum=100, step=5),
        gr.Slider(label="Patch Size", value=64, minimum=32, maximum=120, step=8),
        gr.Textbox(
            label="Seed",
            value=0,
            max_lines=1,
            placeholder="0",
        ),
        gr.Checkbox(label="Low VRAM", value=False),
        gr.Checkbox(label="Tiled Decoder", value=False)


    ],
    outputs=gr.Gallery(columns=2, preview=True, allow_preview=True),
    title="ElasticDiffusion: Training-free Arbitrary Size Image Generation through Global-Local Content Separation",
    description=description,
    article=article,
    examples=[
        ["imgs/input/yoga.jpeg", "Envision a dramatic photo of the joker, masterpiece, high quality.", "blurry, ugly, poorly drawn, deformed", 'canny', 1536, 1536, 7, 0.3, 1000, None, None, None, None, None, None, None, 0, None, None],
    ],
              
    allow_flagging=False,
).launch(server_port=args.port)
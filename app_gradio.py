import time
import gradio as gr
from elastic_diffusion import ElasticDiffusion
import torch
import argparse

loaded_sd_version = 'XL1.0'
device = torch.device('cuda')
pipe = ElasticDiffusion(device, loaded_sd_version, verbose=False)

def generate_image_fn(prompt:str, 
                      negative_prompts:str,
                      img_width:int, 
                      img_height:int=512,
                      resampling_steps:int=20,
                      resampling_keep_p:float=0.8,
                      rrg_scale:float=200,
                      guidance_scale:float=10.0,
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
        
    start_time = time.time()
    pipe.seed_everything(int(seed))
    pipe.view_batch_size = int(view_batch_size)
    pipe.set_view_config(patch_size=patch_size)

    if pipe.low_vram != low_vram:
        pipe = ElasticDiffusion(device, loaded_sd_version, low_vram=True, verbose=False)

    images, log_info = pipe.generate_image(prompts=prompt, negative_prompts=negative_prompts,
                            height=img_height, width=img_width, 
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=guidance_scale, 
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

description = """
 """
article = ""
gr.Interface(
    generate_image_fn,
    inputs=[
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
        gr.Slider(label="Width", value=1024, minimum=128, maximum=2048, step=128),
        gr.Slider(label="Height", value=1024, minimum=128, maximum=2048, step=128),
        gr.Slider(label="Resampling: Steps", value=10, minimum=0, maximum=39, step=3),
        gr.Slider(label="Resampling: Percentage of newly sampled pixels", value=0.3, minimum=0.1, maximum=0.5, step=0.1),
        gr.Slider(label="Reduced Resolution Guidance Scale", value=200, minimum=0, maximum=4000, step=100),
        gr.Slider(label="Classifer-Free Guidance Scale", value=10, minimum=5, maximum=12, step=1),
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
        ["A realistic portrait of a young black woman. she has a Christmas red hat and a red scarf. Her eyes are light brown like they're almost caramel color. Her attire, simple yet dignified.", "blurry, ugly, poorly drawn, deformed", 2048, 2048, 10, 0.3, 2000, None, None, None, None, None, None, 0, None, None],
        ["Envision a portrait of a horse, framed by a blue headscarf with muted tones of rust and cream. she has brown-colored eyes. Her attire, simple yet dignified", "blurry, ugly, poorly drawn, deformed", 1536, 1536, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["Envision a portrait of a cute corgi, framed by a red headscarf. his eyes are light brown. his attire is simple yet dignified", "blurry, ugly, poorly drawn, deformed", 1024, 2048, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["Envision an ostrich in the dessert. she has a green scarf wrapping her body. her eyes are dark black. her attire, simple yet dignified", "blurry, ugly, poorly drawn, deformed", 2048, 1024, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["Envision a portrait of a cute cat, her face is framed by a blue headscarf with muted tones of rust and cream. Her eyes are blue like faded denim. Her attire, simple yet dignified", "blurry, ugly, poorly drawn, deformed", 1080, 1920, 7, 0.3, 1000, None, None, None,None, None, None, 0, None, None],
        ["Envision a realistic portrait of a black woman, she has a white headscarf. Her eyes are dark black. her attire, simple yet delightful.", "blurry, ugly, poorly drawn, deformed", 1920, 1080, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["A Cute Puppy with wings, Cartoon Drawings, high details", "blurry, ugly, poorly drawn, deformed", 2048, 1536, 10, 0.3, 1500, None, None, None, None, None, None, 0, None, None],
        ["Darth Vader playing with raccoon in Mars during sunset.", "blurry, ugly, poorly drawn, deformed", 1536, 2048, 10, 0.3, 1500, None, None, None, None, None, None, 0, None, None],
        ["A dramatic photo of a volcanic eruption, high details, sharp.", "blurry, ugly, poorly drawn, deformed", 768, 2048, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["A photo of the dolomites, highly detailed, sharp", "blurry, ugly, poorly drawn, deformed", 2048, 768, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["A professional photo of a rabbit riding a bike on a street in New York", "blurry, ugly, poorly drawn, deformed", 768, 768, 0, 0.3, 0, None, None, None, None, None, None, 0, None, None],
        ["An illustration of an astronaut riding a horse", "blurry, ugly, poorly drawn, deformed", 512, 512, 0, 0.3, 0, None, None, None, None, None, None, 0, None, None],
        ["A front view of a beautiful waterfall", "blurry, ugly, poorly drawn, deformed", 2048, 512, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None],
        ["A realistic bird-eye view of a lake with palm tree on the side, simply, high details.", "blurry, ugly, poorly drawn, deformed", 512, 2048, 7, 0.3, 1000, None, None, None, None, None, None, 0, None, None]],
              
    allow_flagging=False,
).launch(server_port=args.port)
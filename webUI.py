import gradio as gr
import torch

@torch.no_grad()
def process(input_image, seed):
    return [input_image]
  
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Rerender A Video")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run All") 
            with gr.Row():
                run_button1 = gr.Button(value="Run 1st Key Frame")
                run_button2 = gr.Button(value="Run Key Frames")
                run_button3 = gr.Button(value="Run Propagation") 
            with gr.Accordion("Advanced options for the 1st frame translation", open=False):
                image_resolution = gr.Slider(label="Frame rsolution", minimum=256, maximum=768, value=512, step=64)
                control_strength = gr.Slider(label="ControNet strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                color_preserve = gr.Checkbox(label='Preserve color', value=True)
                with gr.Row():
                    low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                    high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="CFG scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=0, randomize=False)
                a_prompt = gr.Textbox(label="Added prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Accordion("Advanced options for the key fame translation", open=False):
                frame_interval = gr.Slider(label="Key frame frequency", minimum=1, maximum=100, value=10, step=1)
                keyframe_count = gr.Slider(label="Num of key frames", minimum=1, maximum=100, value=10, step=1)       
                x0_strength = gr.Slider(label="Denoising strength", minimum=0.00, maximum=1.05, value=0.75, step=0.05) 
                use_constraints = gr.CheckboxGroup(["shape-aware fusion", "pixel-aware fusion", "color-aware AdaIN"], 
                                                   label="Select the cross-frame contraints to be used",
                                                  value=["shape-aware fusion", "pixel-aware fusion", "color-aware AdaIN"]),                
                with gr.Row():
                    cross_start = gr.Slider(label="Cross-frame attention start", minimum=0, maximum=1, value=0, step=0.05)
                    cross_end = gr.Slider(label="Cross-frame attention end", minimum=0, maximum=1, value=1, step=0.05)  
                with gr.Row():
                    warp_start = gr.Slider(label="Shape-aware fusion start", minimum=0, maximum=1, value=0, step=0.05)
                    warp_end = gr.Slider(label="Shape-aware fusion end", minimum=0, maximum=1, value=0.1, step=0.05)                  
                with gr.Row():
                    mask_start = gr.Slider(label="Pixel-aware fusion start", minimum=0, maximum=1, value=0.5, step=0.05)
                    mask_end = gr.Slider(label="Pixel-aware fusion end", minimum=0, maximum=1, value=0.8, step=0.05)
                with gr.Row():
                    ada_start = gr.Slider(label="Color-aware AdaIN start", minimum=0, maximum=1, value=0.8, step=0.05)
                    ada_end = gr.Slider(label="Color-aware AdaIN end", minimum=0, maximum=1, value=1, step=0.05)                    
                mask_strength = gr.Slider(label="Pixel-aware fusion stength", minimum=0, maximum=1, value=0.5, step=0.01)
                inner_strength = gr.Slider(label="Pixel-aware fusion detail level (lower to prevent artifacts)", minimum=0.5, maximum=1, value=0.9, step=0.01)
                smooth_boundary = gr.Checkbox(label='Smooth fusion boundary (prevent artifacts at boundary)', value=True) 
            with gr.Accordion("Advanced options for the full video translation", open=False):
                MAX_PROCESS = gr.Slider(label="Number of parallel processes", minimum=1, maximum=8, value=8, step=1)

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    
    
block.launch(server_name='0.0.0.0')

    

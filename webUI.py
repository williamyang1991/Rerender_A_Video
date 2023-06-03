import gradio as gr
import torch
import imageio
import cv2

@torch.no_grad()
def process(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS):
    
    first_frame = process1(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS)
    
    keypath = process2(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS)

    fullpath = process3(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS)    

    return first_frame, keypath, fullpath

@torch.no_grad()
def process1(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS):
    video_cap = cv2.VideoCapture(input_video) 
    success, frame = video_cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_cap.release()
    return frame

@torch.no_grad()
def process2(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS):
    path = 'tmp.mp4'
    video_cap = cv2.VideoCapture(input_video) 
    
    fps = video_cap.get(5)
    num = min(video_cap.get(7), (keyframe_count - 1) * frame_interval + 2)
    success, frame = video_cap.read()
    outputs = []
    for i in range(num-1):
        success, frame = video_cap.read()
        if i % frame_interval != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outputs.append(frame)
    video_cap.release()
    imageio.mimsave(path, outputs, fps=max(1, fps//frame_interval))
    return path

@torch.no_grad()
def process3(input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints, cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS):
    path = input_video # 'blend.mp4'
    return path
  
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Rerender A Video")
    with gr.Row():
        with gr.Column():
            #input_image = gr.Image(source='upload', type="numpy")
            input_video = gr.Video(label="Input Video", source='upload', format="mp4", visible=True)
            prompt = gr.Textbox(label="Prompt")
            seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=0, randomize=True)
            run_button = gr.Button(value="Run All") 
            with gr.Row():
                run_button1 = gr.Button(value="Run 1st Key Frame")
                run_button2 = gr.Button(value="Run Key Frames")
                run_button3 = gr.Button(value="Run Propagation") 
            with gr.Accordion("Advanced options for the 1st frame translation", open=False):
                image_resolution = gr.Slider(label="Frame rsolution", minimum=256, maximum=768, value=512, step=64)
                control_strength = gr.Slider(label="ControNet strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                color_preserve = gr.Checkbox(label='Preserve color', value=True, info="Keep the color of the input video")
                with gr.Row():
                    left_crop = gr.Slider(label="Left crop length", minimum=0, maximum=512, value=0, step=1)
                    right_crop = gr.Slider(label="Right crop length", minimum=0, maximum=512, value=0, step=1)
                with gr.Row():
                    top_crop = gr.Slider(label="Top crop length", minimum=0, maximum=512, value=0, step=1)
                    bottom_crop = gr.Slider(label="Bottom crop length", minimum=0, maximum=512, value=0, step=1)
                with gr.Row():
                    control_type = gr.Dropdown(["HED", "canny"], label="Control type", value="HED")
                    low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                    high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="CFG scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1) 
                sd_model = gr.Dropdown(["Stable Diffusion 1.5"], label="Base model", value="Stable Diffusion 1.5")
                a_prompt = gr.Textbox(label="Added prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Accordion("Advanced options for the key fame translation", open=False):
                frame_interval = gr.Slider(label="Key frame frequency (K)", minimum=1, maximum=100, value=10, step=1,
                                          info="Uniformly sample the key frames every K frames")
                keyframe_count = gr.Slider(label="Number of key frames", minimum=1, maximum=100, value=11, step=1)       
                x0_strength = gr.Slider(label="Denoising strength", minimum=0.00, maximum=1.05, value=0.75, step=0.05,
                                       info="0: fully recover the input. 1.05: fully rerender the input.") 
                use_constraints = gr.CheckboxGroup(["shape-aware fusion", "pixel-aware fusion", "color-aware AdaIN"], 
                                                   label="Select the cross-frame contraints to be used",
                                                  value=["shape-aware fusion", "pixel-aware fusion", "color-aware AdaIN"]),                
                with gr.Row():
                    cross_start = gr.Slider(label="Cross-frame attention start", minimum=0, maximum=1, value=0, step=0.05)
                    cross_end = gr.Slider(label="Cross-frame attention end", minimum=0, maximum=1, value=1, step=0.05)  
                style_update_freq = gr.Slider(label="Cross-frame attention update frequency", minimum=1, maximum=100, value=1, step=1,
                                       info="Update the key and value for cross-frame attention every N key frames")                       
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
                inner_strength = gr.Slider(label="Pixel-aware fusion detail level", minimum=0.5, maximum=1, value=0.9, step=0.01,
                                          info="Use a low value to prevent artifacts")
                smooth_boundary = gr.Checkbox(label='Smooth fusion boundary', value=True,
                                             info="select to prevent artifacts at boundary") 
            with gr.Accordion("Advanced options for the full video translation", open=False):
                MAX_PROCESS = gr.Slider(label="Number of parallel processes", minimum=1, maximum=8, value=8, step=1)

        with gr.Column():
            result_image = gr.Image(label='Output first frame', type='numpy', interactive=False)
            #result_gallery = gr.Gallery(label='Output key frames', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_keyframe = gr.Video(label='Output key frame video', format='mp4', interactive=False)  
            result_video = gr.Video(label='Output full video', format='mp4', interactive=False)  
    ips = [input_video, prompt, image_resolution, control_strength, color_preserve, left_crop, right_crop, top_crop, bottom_crop,
           control_type, low_threshold, high_threshold, ddim_steps, scale, seed, sd_model, a_prompt, n_prompt,
          frame_interval, keyframe_count, x0_strength, use_constraints[0], cross_start, cross_end, style_update_freq, warp_start, warp_end,
          mask_start, mask_end, ada_start, ada_end, mask_strength, inner_strength, smooth_boundary, MAX_PROCESS]
    run_button.click(fn=process, inputs=ips, outputs=[result_image, result_keyframe, result_video])
    run_button1.click(fn=process1, inputs=ips, outputs=[result_image])
    run_button2.click(fn=process2, inputs=ips, outputs=[result_keyframe])
    run_button3.click(fn=process3, inputs=ips, outputs=[result_video])
    
    
block.launch(server_name='0.0.0.0')

    

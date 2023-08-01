# Rerender A Video - Official PyTorch Implementation

![Untitled](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/13b8538b-d321-477f-9887-b79e04982da6)

<!--https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/82c35efb-e86b-4376-bfbe-6b69159b8879-->


**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
[**Project Page**](https://www.mmlab-ntu.com/project/rerender/) | [**Paper**](#) | [**Supplementary Video**](#) <br>

<a href="https://huggingface.co/spaces/Anonymous-sub/Rerender"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Web Demo"></a>

> **Abstract:** *Large text-to-image diffusion models have exhibited impressive proficiency in generating high-quality images. However, when applying these models to video domain, ensuring temporal consistency across video frames remains a formidable challenge. This paper proposes a novel zero-shot text-guided video-to-video translation framework to adapt image models to videos. The framework includes two parts: key frame translation and full video translation. The first part uses an adapted diffusion model to generate key frames, with hierarchical cross-frame constraints applied to enforce coherence in shapes, textures and colors. The second part propagates the key frames to other frames with temporal-aware patch matching and frame blending. Our framework achieves global style and local texture temporal consistency at a low cost (without re-training or optimization). The adaptation is compatible with existing image diffusion techniques, allowing our framework to take advantage of them, such as customizing a specific subject with LoRA, and introducing extra spatial guidance with ControlNet. Extensive experimental results demonstrate the effectiveness of our proposed framework over existing methods in rendering high-quality and temporally-coherent videos.*

**Features**:<br>
- **Temporal consistency**: cross-frame constraints for low-level temporal consistency.
- **Zero-shot**: no training or fine-tuning required.
- **Flexibility**: compatible with off-the-shelf models (e.g., [ControlNet](https://github.com/lllyasviel/ControlNet), [LoRA](https://civitai.com/)) for customized translation.

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/7521be69-57c0-499b-859b-86368c14612d

## Updates

- [06/2023] Integrated to 🤗 [Hugging Face](https://huggingface.co/spaces/Anonymous-sub/Rerender). Enjoy the web demo!
- [05/2023] This website is created.

### TODO
- [x] Fix the bug of black boundary during propagation.
- [x] Integrate into Diffusers.
- [x] Add Inference instructions in README.md.
- [x] ~~Add Examples to webUI.~~
- [x] ~~Add optional poisson fusion to the pipeline.~~
- [x] ~~Add Installation instructions for Windows~~

## Installation

1. Clone the repository.

```shell
git clone git@github.com:williamyang1991/Rerender_A_Video.git --recursive
cd Rerender_A_Video
```

2. If you have installed PyTorch CUDA, you can simply set up the environment with pip.

```shell
pip install -r requirements.txt
```

You can also create a new conda environment from scratch.

```shell
conda env create -f environment.yml
conda activate rerender
```

3. Run the installation script. The required models will be downloaded in `./models`.

```shell
python install.py
```

4. You can run the demo with `rerender.py`

```shell
python rerender.py --cfg config/real2sculpture.json
```

<details>
<summary>Installation on Windows</summary>

  Before running the above 1-4 steps, you need prepare:
1. Install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
2. Install [git](https://git-scm.com/download/win)
3. Install [VS](https://visualstudio.microsoft.com/) with Windows 10/11 SDK (for building deps/ebsynth/bin/ebsynth.exe)
</details>

## (1) Inference

### WebUI (recommended)

```
python webUI.py
```
The Gradio app also allows you to flexibly change the inference options. Just try it for more details.

Upload your video, input the prompt, select the seed, and hit:
- **Run 1st Key Frame**: only translate the first frame, so you can adjust the prompts/models/parameters to find your ideal output appearance before running the whole video.
- **Run Key Frames**: translate all the key frames based on the settings of the first frame, so you can adjust the temporal-related parameters for better temporal consistency before running the whole video.
- **Run Propogation**: propogate the key frames to other frames for full video translation
- **Run All**: **Run 1st Key Frame**, **Run Key Frames** and **Run Propogation**

![UI](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/d4d9160d-0990-4397-bf3d-07edcf56a738)

We provide abundant advanced options to play with

<details>
<summary> <b>Using customized models</b></summary>

- Using LoRA/Dreambooth/Finetuned/Mixed SD models
  - Modify `sd_model_cfg.py` to add paths to the saved SD models
- Using other controls from ControlNet (e.g., Depth, Pose)
  - Add more options like `control_type = gr.Dropdown(['HED', 'canny', 'depth']` here https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L690
  - Add model loading options like `elif control_type == 'depth':` following https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L88
  - Add model detectors like `elif control_type == 'depth':` following https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L122
  - One example is given [here](https://huggingface.co/spaces/Anonymous-sub/Rerender/discussions/10/files) 
  
</details>

<details>
<summary> <b>Advanced options for the 1st frame translation</b></summary>

1. Resolution related (**Frame resolution**, **left/top/right/bottom crop length**): crop the frame and resize its short side to 512.
2. ControlNet related:
   - **ControlNet strength**: how well the output matches the input control edges
   - **Control type**: HED edge or Canny edge
   - **Canny low/high threshold**: low values for more edge details
3. SDEdit related:
   - **Denoising strength**: repaint degree (low value to make the output look more like the original video)
   - **Preserve color**: preserve the color of the original video
4. SD related:
   - **Steps**: denoising step
   - **CFG scale**: how well the output matches the prompt
   - **Base model**: base Stable Diffusion model (SD 1.5)
     - Stable Diffusion 1.5: official model
     - [revAnimated_v11](https://civitai.com/models/7371/rev-animated?modelVersionId=19575): a semi-realistic (2.5D) model
     - [realisticVisionV20_v20](https://civitai.com/models/4201?modelVersionId=29460): a photo-realistic model
   - **Added prompt/Negative prompt**: supplementary prompts

</details>

<details>
<summary> <b>Advanced options for the key frame translation</b></summary>

1. Key frame related
   - **Key frame frequency (K)**: Uniformly sample the key frame every K frames. Small value for large or fast motions.
   - **Number of key frames (M)**: The final output video will have K*M+1 frames with M+1 key frames.
2. Temporal consistency related
   - Cross-frame attention: 
     - **Cross-frame attention start/end**: When applying cross-frame attention for global style consistency
     - **Cross-frame attention update frequency (N)**: Update the reference style frame every N key frames. Should be large for long videos to avoid error accumulation.
   - **Shape-aware fusion** Check to use this feature 
     - **Shape-aware fusion start/end**: When applying shape-aware fusion for local shape consistency
   - **Pixel-aware fusion** Check to use this feature 
     - **Pixel-aware fusion start/end**: When applying pixel-aware fusion for pixel-level temporal consistency
     - **Pixel-aware fusion strength**: The strength to preserve the non-inpainting region. Small to avoid error accumulation. Large to avoid burry textures.
     - **Pixel-aware fusion detail level**: The strength to sharpen the inpainting region. Small to avoid error accumulation. Large to avoid burry textures.
     - **Smooth fusion boundary**: Check to smooth the inpainting boundary (avoid error accumulation).
   - **Color-aware AdaIN** Check to use this feature 
     - **Color-aware AdaIN start/end**: When applying AdaIN to make the video color consistent with the first frame

</details>

<details>
<summary> <b>Advanced options for the full video translation</b></summary>
  
1. **Gradient blending**: apply Poisson Blending to reduce ghosting artifats. May slow the process and increase flickers.
2. **Number of parallel processes**: multiprocessing to speed up the process. Large value (8) is recommended.
</details>

![options](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/d133e495-01f1-456f-8c41-0ff319721781)


TODO FOR COMMAND LINE
1. Rerender the first key frame
2. Rerender the full key frames
3. Rerender the full video with propagation
4. Use your own customized model
5. Tips


## (2) Results

### Key frame translation

<table class="center">
<tr>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/cc6db004-8366-4dde-bac0-a2ebd2d23d61" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/1dd74c11-a9c1-4ea9-ba60-45150e5ed3ca" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/542dbc93-e5df-4347-964d-a11c1ae7c9ed" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/58c0afcd-9bcd-4564-9aa5-202502b35f60" raw=true></td>
</tr>
<tr>
  <td width=27.5% align="center">white ancient Greek sculpture, Venus de Milo, light pink and blue background</td>
  <td width=27.5% align="center">a handsome Greek man</td>
  <td width=21.5% align="center">a traditional mountain in chinese ink wash painting</td>
  <td width=23.5% align="center">a cartoon tiger</td>
</tr>
</table>

<table class="center">
<tr>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/ea3d919b-01d8-40f6-b708-48d33beda854" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/932aa462-93a6-44cb-8598-127b1184b53a" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/2fe6ec77-bb68-4d4c-954d-9a2f51a9b975" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/fde7ff9d-7d96-4b22-b5b2-307c6ea2ccc3" raw=true></td>
</tr>
<tr>
  <td width=26.0% align="center">a swan in chinese ink wash painting, monochrome</td>
  <td width=29.0% align="center">a beautiful woman in CG style</td>
  <td width=21.5% align="center">a clean simple white jade sculpture</td>
  <td width=24.0% align="center">a fluorescent jellyfish in the deep dark blue sea</td>
</tr>
</table>

### Full video translation

Text-guided virtual character generation.

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/f0e39ed5-080d-4aec-9a6b-fe81b37b29fe

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/13741837-b065-476b-84fb-772b159f456a

Video stylization and video editing.

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/79503b36-cf61-4e64-b57e-540494d6dd88


## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yang2023rerender,
 title = {Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation},
 author = {Yang, Shuai and Zhou, Yifan and Liu, Ziwei and and Loy, Chen Change},
 year = {2023},
}
```

## Acknowledgments

The code is mainly developed based on [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [GMFlow](https://github.com/haofeixu/gmflow) and [Ebsynth](https://github.com/jamriska/ebsynth).

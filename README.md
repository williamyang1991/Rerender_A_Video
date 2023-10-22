# Rerender A Video - Official PyTorch Implementation

![teaser](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/aa7dc164-dab7-43f4-a46b-758b34911f16)

<!--https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/82c35efb-e86b-4376-bfbe-6b69159b8879-->


**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
in SIGGRAPH Asia 2023 Conference Proceedings <br>
[**Project Page**](https://www.mmlab-ntu.com/project/rerender/) | [**Paper**](https://arxiv.org/abs/2306.07954) | [**Supplementary Video**](https://youtu.be/cxfxdepKVaM) | [**Input Data and Video Results**](https://drive.google.com/file/d/1HkxG5eiLM_TQbbMZYOwjDbd5gWisOy4m/view?usp=sharing) <br>

<a href="https://huggingface.co/spaces/Anonymous-sub/Rerender"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Web Demo"></a> ![visitors](https://visitor-badge.laobi.icu/badge?page_id=williamyang1991/Rerender_A_Video)

> **Abstract:** *Large text-to-image diffusion models have exhibited impressive proficiency in generating high-quality images. However, when applying these models to video domain, ensuring temporal consistency across video frames remains a formidable challenge. This paper proposes a novel zero-shot text-guided video-to-video translation framework to adapt image models to videos. The framework includes two parts: key frame translation and full video translation. The first part uses an adapted diffusion model to generate key frames, with hierarchical cross-frame constraints applied to enforce coherence in shapes, textures and colors. The second part propagates the key frames to other frames with temporal-aware patch matching and frame blending. Our framework achieves global style and local texture temporal consistency at a low cost (without re-training or optimization). The adaptation is compatible with existing image diffusion techniques, allowing our framework to take advantage of them, such as customizing a specific subject with LoRA, and introducing extra spatial guidance with ControlNet. Extensive experimental results demonstrate the effectiveness of our proposed framework over existing methods in rendering high-quality and temporally-coherent videos.*

**Features**:<br>
- **Temporal consistency**: cross-frame constraints for low-level temporal consistency.
- **Zero-shot**: no training or fine-tuning required.
- **Flexibility**: compatible with off-the-shelf models (e.g., [ControlNet](https://github.com/lllyasviel/ControlNet), [LoRA](https://civitai.com/)) for customized translation.

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/811fdea3-f0da-49c9-92b8-2d2ad360f0d6

## Updates
- [10/2023] New features: [Loose cross-frame attention](#loose-cross-frame-attention) and [FreeU](#freeu).
- [09/2023] Code is released.
- [09/2023] Accepted to SIGGRAPH Asia 2023 Conference Proceedings!
- [06/2023] Integrated to 🤗 [Hugging Face](https://huggingface.co/spaces/Anonymous-sub/Rerender). Enjoy the web demo!
- [05/2023] This website is created.

### TODO
- [x] Integrate into Diffusers.
- [x] ~~Integrate [FreeU](https://github.com/ChenyangSi/FreeU) into Rerender~~
- [x] ~~Add Inference instructions in README.md.~~
- [x] ~~Add Examples to webUI.~~
- [x] ~~Add optional poisson fusion to the pipeline.~~
- [x] ~~Add Installation instructions for Windows~~

## Installation

*Please make sure your installation path only contain English letters or _*

1. Clone the repository. (Don't forget --recursive. Otherwise, please run `git submodule update --init --recursive`)

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
24GB VRAM is required. Please refer to https://github.com/williamyang1991/Rerender_A_Video/pull/23#issue-1900789461 to reduce memory consumption.

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
4. [Here](https://github.com/williamyang1991/Rerender_A_Video/issues/18#issuecomment-1752712233) are more information. If building ebsynth fails, we provides our complied [ebsynth](https://drive.google.com/drive/folders/1oSB3imKwZGz69q2unBUfcgmQpzwccoyD?usp=sharing). 
</details>

<details id="issues">
<summary>🔥🔥🔥 <b>Installation or Running Fails?</b> 🔥🔥🔥</summary>

1. In case building ebsynth fails, we provides our complied [ebsynth](https://drive.google.com/drive/folders/1oSB3imKwZGz69q2unBUfcgmQpzwccoyD?usp=sharing)
2. `FileNotFoundError: [Errno 2] No such file or directory: 'xxxx.bin' or 'xxxx.jpg'`:
    - make sure your path only contains English letters or _ (https://github.com/williamyang1991/Rerender_A_Video/issues/18#issuecomment-1723361433)
    - find the code `python video_blend.py ...` in the error log and use it to manually run the ebsynth part, which is more stable than WebUI.
    - if some non-keyframes are generated but somes are not, rather than missing all non-keyframes in '/out_xx/', you may refer to https://github.com/williamyang1991/Rerender_A_Video/issues/38#issuecomment-1730668991
5. `KeyError: 'dataset'`: upgrade Gradio to the latest version (https://github.com/williamyang1991/Rerender_A_Video/issues/14#issuecomment-1722778672, https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11855)
6. Error when processing videos: manually install ffmpeg (https://github.com/williamyang1991/Rerender_A_Video/issues/19#issuecomment-1723685825, https://github.com/williamyang1991/Rerender_A_Video/issues/29#issuecomment-1726091112)
7. `ERR_ADDRESS_INVALID` Cannot open the webUI in browser: replace 0.0.0.0 with 127.0.0.1 in webUI.py (https://github.com/williamyang1991/Rerender_A_Video/issues/19#issuecomment-1723685825)
8. `CUDA out of memory`:
     - Using xformers (https://github.com/williamyang1991/Rerender_A_Video/pull/23#issue-1900789461)
     - Set `"use_limit_device_resolution"` to `true` in the config to resize the video according to your VRAM (https://github.com/williamyang1991/Rerender_A_Video/issues/79). An example config `config/van_gogh_man_dynamic_resolution.json` is provided.
10. `AttributeError: module 'keras.backend' has no attribute 'is_tensor'`: update einops (https://github.com/williamyang1991/Rerender_A_Video/issues/26#issuecomment-1726682446)
11. `IndexError: list index out of range`: use the original DDIM steps of 20 (https://github.com/williamyang1991/Rerender_A_Video/issues/30#issuecomment-1729039779)
12. One-click installation https://github.com/williamyang1991/Rerender_A_Video/issues/99

</details>


## (1) Inference

### WebUI (recommended)

```
python webUI.py
```
The Gradio app also allows you to flexibly change the inference options. Just try it for more details. (For WebUI, you need to download [revAnimated_v11](https://civitai.com/models/7371/rev-animated?modelVersionId=19575) and [realisticVisionV20_v20](https://civitai.com/models/4201?modelVersionId=29460) to `./models/` after Installation)

Upload your video, input the prompt, select the seed, and hit:
- **Run 1st Key Frame**: only translate the first frame, so you can adjust the prompts/models/parameters to find your ideal output appearance before running the whole video.
- **Run Key Frames**: translate all the key frames based on the settings of the first frame, so you can adjust the temporal-related parameters for better temporal consistency before running the whole video.
- **Run Propagation**: propagate the key frames to other frames for full video translation
- **Run All**: **Run 1st Key Frame**, **Run Key Frames** and **Run Propagation**

![UI](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/eb4e1ddc-11a3-42dd-baa4-622eecef04c7)


We provide abundant advanced options to play with

<details id="option0">
<summary> <b>Using customized models</b></summary>

- Using LoRA/Dreambooth/Finetuned/Mixed SD models
  - Modify `sd_model_cfg.py` to add paths to the saved SD models
  - How to use LoRA: https://github.com/williamyang1991/Rerender_A_Video/issues/39#issuecomment-1730678296
- Using other controls from ControlNet (e.g., Depth, Pose)
  - Add more options like `control_type = gr.Dropdown(['HED', 'canny', 'depth']` here https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L690
  - Add model loading options like `elif control_type == 'depth':` following https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L88
  - Add model detectors like `elif control_type == 'depth':` following https://github.com/williamyang1991/Rerender_A_Video/blob/b6cafb5d80a79a3ef831c689ffad92ec095f2794/webUI.py#L122
  - One example is given [here](https://huggingface.co/spaces/Anonymous-sub/Rerender/discussions/10/files)

</details>

<details id="option1">
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
5. FreeU related:
   - **FreeU first/second-stage backbone factor**: =1 do nothing; >1 enhance output color and details
   - **FreeU first/second-stage skip factor**: =1 do nothing; <1 enhance output color and details

</details>

<details id="option2">
<summary> <b>Advanced options for the key frame translation</b></summary>

1. Key frame related
   - **Key frame frequency (K)**: Uniformly sample the key frame every K frames. Small value for large or fast motions.
   - **Number of key frames (M)**: The final output video will have K*M+1 frames with M+1 key frames.
2. Temporal consistency related
   - Cross-frame attention:
     - **Cross-frame attention start/end**: When applying cross-frame attention for global style consistency
     - **Cross-frame attention update frequency (N)**: Update the reference style frame every N key frames. Should be large for long videos to avoid error accumulation.
     - **Loose Cross-frame attention**: Using cross-frame attention in fewer layers to better match the input video (for video with large motions)
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

<details id="option3">
<summary> <b>Advanced options for the full video translation</b></summary>

1. **Gradient blending**: apply Poisson Blending to reduce ghosting artifacts. May slow the process and increase flickers.
2. **Number of parallel processes**: multiprocessing to speed up the process. Large value (8) is recommended.
</details>

![options](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/ffebac15-e7e0-4cd4-a8fe-60f243450172)


### Command Line

We also provide a flexible script `rerender.py` to run our method.

#### Simple mode

Set the options via command line. For example,

```shell
python rerender.py --input videos/pexels-antoni-shkraba-8048492-540x960-25fps.mp4 --output result/man/man.mp4 --prompt "a handsome man in van gogh painting"
```

The script will run the full pipeline. A work directory will be created at `result/man` and the result video will be saved as `result/man/man.mp4`

#### Advanced mode

Set the options via a config file. For example,

```shell
python rerender.py --cfg config/van_gogh_man.json
```

The script will run the full pipeline.
We provide some examples of the config in `config` directory.
Most options in the config is the same as those in WebUI.
Please check the explanations in the WebUI section.

Specifying customized models by setting `sd_model` in config. For example:
```json
{
  "sd_model": "models/realisticVisionV20_v20.safetensors",
}
```

#### Customize the pipeline

Similar to WebUI, we provide three-step workflow: Rerender the first key frame, then rerender the full key frames, finally rerender the full video with propagation. To run only a single step, specify options `-one`, `-nb` and `-nr`:

1. Rerender the first key frame
```shell
python rerender.py --cfg config/van_gogh_man.json -one -nb
```
2. Rerender the full key frames
```shell
python rerender.py --cfg config/van_gogh_man.json -nb
```
3. Rerender the full video with propagation
```shell
python rerender.py --cfg config/van_gogh_man.json -nr
```

#### Our Ebsynth implementation

We provide a separate Ebsynth python script `video_blend.py` with the temporal blending algorithm introduced in
[Stylizing Video by Example](https://dcgi.fel.cvut.cz/home/sykorad/ebsynth.html) for interpolating style between key frames.
It can work on your own stylized key frames independently of our Rerender algorithm.

Usage:
```shell
video_blend.py [-h] [--output OUTPUT] [--fps FPS] [--beg BEG] [--end END] [--itv ITV] [--key KEY]
                      [--n_proc N_PROC] [-ps] [-ne] [-tmp]
                      name

positional arguments:
  name             Path to input video

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  Path to output video
  --fps FPS        The FPS of output video
  --beg BEG        The index of the first frame to be stylized
  --end END        The index of the last frame to be stylized
  --itv ITV        The interval of key frame
  --key KEY        The subfolder name of stylized key frames
  --n_proc N_PROC  The max process count
  -ps              Use poisson gradient blending
  -ne              Do not run ebsynth (use previous ebsynth output)
  -tmp             Keep temporary output
```
For example, to run Ebsynth on video `man.mp4`,
1. Put the stylized key frames to `videos/man/keys` for every 10 frames (named as `0001.png`, `0011.png`, ...)
2. Put the original video frames in `videos/man/video` (named as `0001.png`, `0002.png`, ...).
3. Run Ebsynth on the first 101 frames of the video with poisson gradient blending and save the result to `videos/man/blend.mp4` under FPS 25 with the following command:
```shell
python video_blend.py videos/man \
  --beg 1 \
  --end 101 \
  --itv 10 \
  --key keys \
  --output videos/man/blend.mp4 \
  --fps 25.0 \
  -ps
```

## (2) Results

### Key frame translation


<table class="center">
<tr>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/18666871-f273-44b2-ae67-7be85d43e2f6" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/61f59540-f06e-4e5a-86b6-1d7cb8ed6300" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/8e8ad51a-6a71-4b34-8633-382192d0f17c" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/b03cd35f-5d90-471a-9aa9-5c7773d7ac39" raw=true></td>
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
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/649a789e-0c41-41cf-94a4-0d524dcfb282" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/73590c16-916f-4ee6-881a-44a201dd85dd" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/fbdc0b8e-6046-414f-a37e-3cd9dd0adf5d" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/eb11d807-2afa-4609-a074-34300b67e6aa" raw=true></td>
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


https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/1405b257-e59a-427f-890d-7652e6bed0a4


https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/efee8cc6-9708-4124-bf6a-49baf91349fc


Video stylization and video editing.


https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/1b72585c-99c0-401d-b240-5b8016df7a3f

## New Features

Compared to the conference version, we are keeping adding new features.

![new_feature](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/98f39f3d-3dfe-4de4-a1b6-99a3c78b5336)

#### Loose cross-frame attention
By using cross-frame attention in less layers, our results will better match the input video, thus reducing ghosting artifacts caused by inconsistencies. This feature can be activated by checking `Loose Cross-frame attention` in the <a href="#option2">Advanced options for the key frame translation</a> for WebUI or setting `loose_cfattn` for script (see `config/real2sculpture_loose_cfattn.json`).

#### FreeU
[FreeU](https://github.com/ChenyangSi/FreeU) is a method that improves diffusion model sample quality at no costs. We find featured with FreeU, our results will have higher contrast and saturation, richer details, and more vivid colors. This feature can be used by setting FreeU backbone factors and skip factors in the <a href="#option1">Advanced options for the 1st frame translation</a> for WebUI or setting `freeu_args` for script (see `config/real2sculpture_freeu.json`).

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang2023rerender,
 title = {Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation},
 author = {Yang, Shuai and Zhou, Yifan and Liu, Ziwei and and Loy, Chen Change},
 booktitle = {ACM SIGGRAPH Asia Conference Proceedings},
 year = {2023},
}
```

## Acknowledgments

The code is mainly developed based on [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [GMFlow](https://github.com/haofeixu/gmflow) and [Ebsynth](https://github.com/jamriska/ebsynth).

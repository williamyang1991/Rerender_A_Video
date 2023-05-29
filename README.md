# Rerender A Video - Official PyTorch Implementation

![Untitled](https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/13b8538b-d321-477f-9887-b79e04982da6)

<!--https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/82c35efb-e86b-4376-bfbe-6b69159b8879-->


**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
[**Project Page**](https://www.mmlab-ntu.com/project/rerender/) | [**Paper**](#) | [**Supplementary Video**](#) <br>

> **Abstract:** *Large text-to-image diffusion models have exhibited impressive proficiency in generating high-quality images. However, when applying these models to video domain, ensuring temporal consistency across video frames remains a formidable challenge. This paper proposes a novel zero-shot text-guided video-to-video translation framework to adapt image models to videos. The framework includes two parts: key frame translation and full video translation. The first part uses an adapted diffusion model to generate key frames, with hierarchical cross-frame constraints applied to enforce coherence in shapes, textures and colors. The second part propagates the key frames to other frames with temporal-aware patch matching and frame blending. Our framework achieves global style and local texture temporal consistency at a low cost (without re-training or optimization). The adaptation is compatible with existing image diffusion techniques, allowing our framework to take advantage of them, such as customizing a specific subject with LoRA, and introducing extra spatial guidance with ControlNet. Extensive experimental results demonstrate the effectiveness of our proposed framework over existing methods in rendering high-quality and temporally-coherent videos.*

**Features**:<br>
- **Temporal consistency**: cross-frame constraints for low-level temporal consistency.
- **Zero-shot**: no training or fine-tuning required.
- **Flexibility**: compatible with off-the-shelf models (e.g., [ControlNet](https://github.com/lllyasviel/ControlNet), [LoRA](https://civitai.com/)) for customized translation.

## Updates

- [05/2023] This website is created.

## Installation

## (1) Inference

## (2) Results


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

The code is mainly developed based on [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

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

https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/7521be69-57c0-499b-859b-86368c14612d

## Updates

- [05/2023] This website is created.

## Installation

## (1) Inference

## (2) Results

### Key frame translation

<table class="center">
<tr>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/4989d95f-4fd0-4777-b918-51fd0bcf318a" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/394dcbeb-f056-4731-9918-6af0d8d16596" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/09425268-d4ae-43f6-87f8-b3a57e322bfe" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/e306ed0a-66da-4ddf-8b98-55c0e0cd7ac7" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/f0f7aa12-b280-4cf9-af4e-5e7d53104b5f" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/d0e2c931-194f-4d91-b389-a725f020eb3b" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/adc77a95-53ab-4d4e-849c-fb12c8855aca" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/e7094f81-71ae-4837-ac3f-bfc43530b0ae" raw=true></td>
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
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/e5315750-9563-4d90-9e04-6514d503cbc5" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/00ed3ff1-6844-4201-a43f-b5ebb437165c" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/a596b5c8-3dc2-4e67-80a9-1ed0b4ac3048" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/12209312-f3cd-4c98-95bd-0b46d0993c01" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/9957da0d-d76a-41c8-9f8f-c8a17fd3c243" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/017c2256-bd06-416e-894a-c60cda6048e7" raw=true></td>
  <td><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/f1d0cb1b-33f4-45c4-8c68-f78542ef8036" raw=true><img src="https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/4e3bdf8a-0c9a-4b5c-877e-70a901a970f0" raw=true></td>
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

The code is mainly developed based on [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

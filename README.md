# ComfyUI_OmniGen_Wrapper

![image](image/omnigen_wrapper_example.jpg)

This node is an unofficial wrapper of the [OmniGen](https://github.com/VectorSpaceLab/OmniGen), running in ComfyUI.
The quantization code is from [Manni1000/OmniGen](https://github.com/Manni1000/OmniGen).
And the base node idea was from [chflame163/ComfyUI_OmniGen_Wrapper](https://github.com/chflame163/ComfyUI_OmniGen_Wrapper).


## Introduction

- OmniGen is an interesting model because it can do various tasks at once.
- It isn't fast
- It isn't high quality. You might want to refine the output using a better model like Flux.
- It consumes plenty of VRAM, in my tests 6 GB were enough to play with the model, but as you add more images and
  increase their size the memory consumption scales.
- This an LLM (SLM in according to Microsoft) based on [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)
  I can handle images mixed with the LLM tokes.

## Installation

### Install the nodes

Open a terminal window in the ```ComfyUI/custom_nodes``` folder and enter the following command:

```
git clone https://github.com/set-soft/ComfyUI_OmniGen_Nodes.git
```

### Install dependencies

Run the following command in the Python environment of ComfyUI:

```
python -s -m pip install -r ComfyUI/custom_nodes/ComfyUI_OmniGen_Nodes/requirements.txt
```

### Download models

I recommend downloading the models by hand, is much faster:

- You need the OmniGen-v1 model.
  - I recommend downloading the FP8 version of the model from
  [silveroxides/OmniGen-V1](https://huggingface.co/silveroxides/OmniGen-V1/tree/main). This is a 3.88 GB file.
  Just download the [model-fp8_e4m3fn.safetensors](https://huggingface.co/silveroxides/OmniGen-V1/resolve/main/model-fp8_e4m3fn.safetensors)
  file.
  - Create a folder named ```ComfyUI/models/OmniGen/``` and move the file there.
  - If you want to use the original model just download it from
    [Shitao/OmniGen-v1](https://huggingface.co/Shitao/OmniGen-v1/tree/main). This version is 15.5 GB and I couldn't find a
    difference in the results. Just download the
    [model.safetensors](https://huggingface.co/Shitao/OmniGen-v1/resolve/main/model.safetensors) file.
    You can move it to the same folder I recommend for the FP8 version.

- You need the SDXL VAE.
  - If you are already using SDXL this should be in ```ComfyUI/models/vae```
  - If you don't have it download the [diffusion_pytorch_model.safetensors](https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors) file
    from [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae/tree/main).
  - Move the file to ```ComfyUI/models/vae```, I suggest renaming it to something like ```sdxl_vae.safetensors```


You don't need the JSON files.

## How to use

The examples use the prompt node from [Comfyroll](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/).
If you have the nodes manager extension just install the missing nodes from the manager.
Otherwise manually install the Comfyroll nodes.

### Very simple demo

1. Drag and drop the following image in ComfyUI:

![image](image/girl_demo_1024.png)

2. Adjust the path for the model in **Omnigen Loader (set)** node
3. Adjust the path for the VAE in **Load VAE** node

Now you can play with this very simple workflow.
The above image was generated using [this image](https://github.com/VectorSpaceLab/OmniGen/blob/main/imgs/demo_cases/t2i_woman_with_book.png)
as input.

### Node Options

![image](image/omnigen_wrapper_node.jpg)

* image_1: Optional input image_1. If input, this image must be described in the prompt and referred to as ```{image_1}```.
* image_2: Optional input image_2. If input, this image must be described in the prompt and referred to as ```{image_2}```.
* image_3: Optional input image_3. If input, this image must be described in the prompt and referred to as ```{image_3}```.
* dtype: Model accuracy, default is the default model accuracy, optional int8. The default precision occupies approximately 12GB of video memory, while int8 occupies approximately 7GB of video memory.
* prompt: The prompt or prompts to guide the image generation. If have image input, use the placeholder ```{image_1}```, ```{image_2}```, ```{image_3}``` to refer to it.
* width: The height in pixels of the generated image. The number must be a multiple of 16.
* height: The width in pixels of the generated image. The number must be a multiple of 16.
* guidance_scale: A higher value will make the generated results of the model more biased towards the condition, but may sacrifice the diversity and degrees of freedom of the image.
* image_guidance_scale: The guidance scale of image.
* steps: The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
* separate_cfg_infer: Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
* use_kv_cache: Enable kv cache to speed up the inference
* seed: A random seed for generating output.
* control_after_generate: Seed value change option every time it runs.
* cache_model: When set to True, the model is cached and does not need to be loaded again during the next run. Strongly recommended for playing with this node.
* move_to_ram: When set to True, keep in VRAM only the needed models. Moves to main RAM the rest. Use it if you experiments issues after inference when the VAE decodes the resulting image.

## Statement

This project follows the MIT license, Some of its functional code comes from other open-source projects.
Thanks to the original author. If used for commercial purposes, please refer to the original project license to authorization
agreement.

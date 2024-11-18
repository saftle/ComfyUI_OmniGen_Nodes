# Python modules
import logging
import os
import sys

# ML modules
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Comfy_UI modules
import folder_paths
import model_management

sys.path.append(os.path.dirname(__file__))
from .omnigen_wrappers import OmniGenProcessorWrapper, OmniGenPipelineWrapper, OmniGenWrapper
from .OmniGen import OmniGenPipeline, OmniGen
from .OmniGen.utils import show_shape, crop_arr, NEGATIVE_PROMPT

r1 = [[0, 1], [1, 0]]
g1 = [[1, 0], [0, 1]]
b1 = [[1, 1], [0, 0]]
EMPTY_IMG = torch.tensor([r1, g1, b1]).unsqueeze(0)
# OmniGen is a Phy-3 based model, technically an SLM model, so I agree this should be stored in:
# <MODELS>/LLM/OmniGen-v1/
# Currently Comfy_UI doesn't define LLM, so here we add it
if not 'LLM' in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["OmniGen"] = ([os.path.join(folder_paths.models_dir, "OmniGen"),
                                                       os.path.join(folder_paths.models_dir, "LLM")], {'.safetensors'})


def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def validate_image(idx, image, prompt, max_input_image_size):
    """ Ensure is used in the prompt, replace by the real marker and resize to a multiple of 16 """
    # Replace {image_N}, optionaly image_N, stop if not in prompt
    img_txt = f"image_{idx}"
    img_txt_curly = "{"+img_txt+"}"
    img_marker = f"<img><|image_{idx}|></img>"
    if img_txt_curly in prompt:
        prompt = prompt.replace(img_txt_curly, img_marker)
    else:
        assert img_txt in prompt, f"Image slot {idx} used, but the image isn't mentioned in the prompt"
        prompt = prompt.replace(img_txt, img_marker)
    # Make the image size usable [B,H,W,C]
    w = image.size(-2)
    h = image.size(-3)
    if w<128 or h<128 or w>max_input_image_size or h>max_input_image_size or w%16 or h%16:
        # Ok, the image needs size adjust
        img = tensor2pil(image)
        img = crop_arr(img, max_input_image_size)
        to_tens = transforms.ToTensor()  # [C,H,W]
        image = to_tens(img).unsqueeze(0).movedim(1, -1)
        logging.info(f"Rescaling image {idx} from {w}x{h} to {image.size(-2)}x{image.size(-3)}")
        logging.debug(image.shape)
    return image, prompt


class OmniGenConditioner:
    def __init__(self):
        self.NODE_NAME = "OmniGen Conditioner"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "input image as {image_1}, e.g.", "multiline":True, "defaultInput": True
                }),
                "max_input_image_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 16
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True, "defaultInput": True}),
            }
        }

    RETURN_TYPES = ("OMNI_COND", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("conditioner", "crp_img_1", "crp_img_2", "crp_img_3")
    FUNCTION = "run"
    CATEGORY = 'OmniGen'

    def run(self, prompt, max_input_image_size, image_1=None, image_2=None, image_3=None, negative=None):

        input_images = []
        if image_1 is not None:
            crp_img_1, prompt = validate_image(1, image_1, prompt, max_input_image_size)
            input_images.append(crp_img_1)
        else:
            crp_img_1 = EMPTY_IMG
        if image_2 is not None:
            assert image_1 is not None, "Don't use image slot 2 if slot 1 is empty"
            crp_img_2, prompt = validate_image(2, image_2, prompt, max_input_image_size)
            input_images.append(crp_img_2)
        else:
            crp_img_2 = EMPTY_IMG
        if image_3 is not None:
            assert image_2 is not None, "Don't use image slot 3 if slot 2 is empty"
            crp_img_3, prompt = validate_image(3, image_3, prompt, max_input_image_size)
            input_images.append(crp_img_3)
        else:
            crp_img_3 = EMPTY_IMG
        if len(input_images) == 0:
            input_images = None

        if negative is None:
            negative = NEGATIVE_PROMPT

        return ({'positive': prompt, 'negative': negative, 'images': input_images}, crp_img_1, crp_img_2, crp_img_3,)


class OmniGenProcessor:
    def __init__(self):
        self.NODE_NAME = "OmniGen Processor"
        self.processor = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition_1": ("OMNI_COND",),
                "separate_cfg_infer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Saves memory and in some cases is even faster"
                }),
                "size_from_first_image": ("BOOLEAN", {
                    "default": True, "tooltip": "Output size will be the same of the first image"
                }),
                "width": ("INT", {
                    "default": 512, "min": 16, "max": 2048, "step": 16,
                    "tooltip": "Width of the output image, unless size_from_first_image is enabled",
                }),
                "height": ("INT", {
                    "default": 512, "min": 16, "max": 2048, "step": 16,
                    "tooltip": "Height of the output image, unless size_from_first_image is enabled",
                }),
            },
            "optional": {
                "condition_2": ("OMNI_COND",),
                "condition_3": ("OMNI_COND",),
            }
        }

    RETURN_TYPES = ("OMNI_FULL_COND",)
    RETURN_NAMES = ("conditioner", )
    FUNCTION = "run"
    CATEGORY = 'OmniGen'

    def run(self, condition_1, separate_cfg_infer, size_from_first_image, width, height, condition_2=None, condition_3=None):
        positive = [condition_1['positive']]
        negative = [condition_1['negative']]
        images = [condition_1['images']]

        if condition_2 is not None:
            positive.append(condition_2['positive'])
            negative.append(condition_2['negative'])
            images.append(condition_2['images'])

        if condition_3 is not None:
            positive.append(condition_3['positive'])
            negative.append(condition_3['negative'])
            images.append(condition_3['images'])

        found_images = False
        final_images = []
        for img in images:
            if img is None:
                final_images.append([])
            else:
                found_images = True
                final_images.append(img)
        if not found_images:
            final_images = None

        if size_from_first_image:
            assert final_images is not None, "Asking to use the size of the first image, but no images provided"
            for imgs in final_images:
                if len(imgs):
                    img = imgs[0]
                    break
            # Images are in Comfy_UI format [B,H,W,C]
            width = img.size(-2)
            height = img.size(-3)

        if not self.processor:
            self.processor = OmniGenProcessorWrapper.from_pretrained()

        input_data = self.processor(positive, final_images, height=height, width=width, use_img_cfg=final_images is not None,
                                    separate_cfg_input=separate_cfg_infer, negative_prompt=negative)

        input_data['separate_cfg_infer'] = separate_cfg_infer
        input_data['input_images'] = final_images
        input_data['num_conditions'] = len(positive)
        input_data['height'] = height
        input_data['width'] = width
        return (input_data,)


class OmniGenSampler:
    def __init__(self):
        self.NODE_NAME = "OmniGen Sampler"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "SDXL VAE to encode the images"}),
                "model": ("OMNI_MODEL", {"tooltip": "OmniGen V1 model"}),
                "conditioner": ("OMNI_FULL_COND",),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1
                }),
                "img_guidance_scale": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 25, "min": 1, "max": 100, "step": 1
                }),
                "use_kv_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable K/V cache to speed up the inference, but slows down the convergence. Needs CUDA"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 1e18, "step": 1
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = 'OmniGen'

    def run(self, vae, model, conditioner, guidance_scale, img_guidance_scale, steps, use_kv_cache, seed):
        # Check the target device
        device = model_management.get_torch_device()
        logging.info(f"Using {device} for the model")
        assert not use_kv_cache or device.startswith('cuda'), "`use_kv_cache` is implemented only for CUDA"
        # Check if using BF16 is convenient
        res = model_management.should_use_bf16(device, 3.8e9)
        dtype = torch.bfloat16 if res else torch.float32
        dtype_s = 'BF16' if res else 'FP32'
        logging.info(f"Using {dtype_s} for the model")
        # Create the pipeline
        self.pipe = OmniGenPipelineWrapper.from_pretrained(model, dtype=dtype, device=device)
        # Generate image
        output = self.pipe(conditioner,
                           num_inference_steps=steps,
                           guidance_scale=guidance_scale,
                           img_guidance_scale=img_guidance_scale,
                           use_kv_cache=use_kv_cache,
                           seed=seed,
                           vae = vae,)

        return ({'samples': output},)


class OmniGenLoader:
    def __init__(self):
        self.NODE_NAME = "OmniGen Loader"
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (folder_paths.get_filename_list("OmniGen"), ),
                              "weight_dtype": (["int8", "default"],)
                             }}
    RETURN_TYPES = ("OMNI_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "OmniGen"

    def load_model(self, name, weight_dtype):
        quantize = weight_dtype == "int8"
        if self.model is None or self.model.quantized != quantize:
            logging.info(f"Loading OmniGen Model")
            # Paths
            fname = folder_paths.get_full_path('OmniGen', name)
            cfg_name = os.path.join(os.path.dirname(__file__), 'model')
            self.model = OmniGenWrapper.from_pretrained(cfg_name, fname, quantize=quantize)
        return (self.model,)


NODE_CLASS_MAPPINGS = {
    "setOmniGenConditioner": OmniGenConditioner,
    "setOmniGenProcessor": OmniGenProcessor,
    "setOmniGenSampler": OmniGenSampler,
    "setOmniGenLoader": OmniGenLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "setOmniGenConditioner": "OmniGen Conditioner (set)",
    "setOmniGenProcessor": "OmniGen Processor (set)",
    "setOmniGenSampler": "OmniGen Sampler (set)",
    "setOmniGenLoader": "OmniGen Loader (set)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

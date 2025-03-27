import argparse
import logging
import math
import os

import PIL.Image
os.environ['HF_DATASETS_CACHE']="/mnt/store/jparanj1/.cache/"
os.environ['TRANSFORMERS_CACHE']='/mnt/store/jparanj1/.cache/'
os.environ['HF_HOME']="/mnt/store/jparanj1/.cache/"
os.environ['HF_HUB_CACHE']='/mnt/store/jparanj1/.cache/'
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import diffusers
import numpy as np
import PIL
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInstructPix2PixPipeline
from gsampipeline import StableDiffusionInstructPix2PixGSAMPipeline
from diffusers.utils import make_image_grid, load_image
from utils import *

pretrained_model_name_or_path = sys.argv[1]
save_name = sys.argv[2]
dataset_name = sys.argv[3]

num_inference_steps = 100
return_text = True
return_boxes = False
return_masks = True
# Load scheduler, tokenizer and models.

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
    revision=None,
)

ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
)

#added linear projection
added_linear = torch.nn.Linear(1024, 768)
added_linear.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path,'added_linear.pth')))
pipeline = StableDiffusionInstructPix2PixGSAMPipeline(
    unet = unet,
    added_linear = added_linear,
    return_boxes=return_boxes,
    return_masks=return_masks,
    return_text=return_text
)

#get list of images for validation
images, gts, names = get_val_images(dataset_name)
preds = []
save_path = os.path.join("predictions", dataset_name, save_name)
os.makedirs(save_path, exist_ok=True)
for i,im in enumerate(images):
    if os.path.exists(os.path.join(save_path, names[i])):
        continue
    init_image = load_image(im)
    with torch.autocast(
                        'cuda',
                        enabled=True,
    ):
        pred_im = pipeline("Create an infrared version of the given image. Make it long wave infrared", image=init_image, im_path=im, num_inference_steps=num_inference_steps).images[0]

    pred_im.save(os.path.join(save_path, names[i]))
    # preds.append(os.path.join(save_path, names[i]))
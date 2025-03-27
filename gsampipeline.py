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

import sys
sys.path.append('/mnt/store/jparanj1/Grounded-Segment-Anything')
sys.path.append("/mnt/store/jparanj1/Grounded-Segment-Anything/GroundingDINO")
sys.path.append("/mnt/store/jparanj1/Grounded-Segment-Anything/recognize-anything")
from transformers import CLIPTextModel, CLIPTokenizer
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer

#gsam requirements
# Grounding DINO
from GroundingDINO.groundingdino.datasets import transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

import accelerate
import cv2
import datasets
import diffusers
import torchvision
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
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import make_image_grid, load_image
from matplotlib import pyplot as plt
from utils import *

def tokenize_captions(tokenizer, captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image_pil)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def get_gsam_outputs(visible_img_path, model, ram_model, sam_predictor,
                     text_encoder, tokenizer, 
                     box_threshold, text_threshold, iou_threshold, 
                     return_text, return_boxes, return_masks, device):
    #get GSAM outputs
    with torch.no_grad():
        image_pil, image = load_image(visible_img_path)
        # image_pil.save("tmp.png")

        # initialize Recognize Anything Model
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
                        TS.Resize((384, 384)),
                        TS.ToTensor(), normalize
                    ])
        
        raw_image = image_pil.resize(
                        (384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device)

        res = inference_ram(raw_image , ram_model)
        tags=res[0].replace(' |', ',')
        # print("predicted tags: ", tags)
        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, tags, box_threshold, text_threshold, device=device
        )

        image = cv2.imread(visible_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image)
        p_enc_2d = PositionalEncodingPermute2D(256).to(device)


        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # print(f"After NMS: {boxes_filt.shape[0]} boxes and total shape: {boxes_filt.shape}")
        # print("pred_phrases: ", pred_phrases)
        selected_tags = [p[:p.find('(')] for p in pred_phrases]
        # print("selected tags: ", selected_tags)
        if return_text:
            text_embeddings = text_encoder(tokenize_captions(tokenizer, selected_tags).to(device))[1]
            # print("text embeddings shape: ", text_embeddings.shape)
        else:
            tmp = text_encoder(tokenize_captions(tokenizer, selected_tags).to(device))[1]
            text_embeddings = torch.zeros_like(tmp).to(device)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        if return_boxes:
            box_embeddings, _ = sam_predictor.model.prompt_encoder(
                points = None,
                boxes = transformed_boxes,
                masks = None
            )
            #legacy
            # box_embeddings = box_embeddings.to(device).reshape((boxes_filt.shape[0],-1))
            #new
            box_embeddings = box_embeddings.to(device)
            #  print("box embeddings shape: ", box_embeddings.shape)
        else:
            if return_masks:
                box_embeddings = None
            else:
                tmp, _ = sam_predictor.model.prompt_encoder(
                    points = None,
                    boxes = transformed_boxes,
                    masks = None
                )
                box_embeddings = torch.zeros_like(tmp).to(device)

        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        if return_masks:
            _, mask_embeddings = sam_predictor.model.prompt_encoder(
                points = None,
                boxes = None,
                masks = masks.float()
            )
            mask_embeddings = mask_embeddings.to(device)
            print("mask embeddings shape: ", mask_embeddings.shape)

            #new
            mask_embeddings_with_pos_embed = mask_embeddings + (p_enc_2d(mask_embeddings))
            # print("mask embeddings with positional embeddings shape: ", mask_embeddings_with_pos_embed.shape)

            #average pool to 100 tokens
            b, c, h, w = mask_embeddings_with_pos_embed.shape
            mask_embeddings_with_pos_embed_avgpooled = torch.nn.functional.adaptive_avg_pool2d(mask_embeddings_with_pos_embed, (10,10))
            # print("mask embeddings with pos embed shape: ", mask_embeddings_with_pos_embed_avgpooled.shape)
            mask_embeddings = mask_embeddings_with_pos_embed_avgpooled.permute(0,2,3,1).reshape((b,100,c))
            # print("mask embeddings shape: ", mask_embeddings.shape)
            mask_embeddings = mask_embeddings.to(device)
        else:
            mask_embeddings = None

    # return natural_im, thermal_im, text_ins, text_embeddings, box_embeddings, mask_embeddings
    return text_embeddings, box_embeddings, mask_embeddings

class StableDiffusionInstructPix2PixGSAMPipeline():
    def __init__(self, unet, added_linear,
                 config_file='/mnt/store/jparanj1/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
                 grounded_checkpoint='/mnt/store/jparanj1/Grounded-Segment-Anything/groundingdino_swint_ogc.pth', 
                 ram_checkpoint='/mnt/store/jparanj1/Grounded-Segment-Anything/ram_swin_large_14m.pth',
                 sam_checkpoint='/mnt/store/jparanj1/sam_vit_b_01ec64.pth',
                 return_boxes=False,
                 return_masks=True,
                 return_text=True,
                 device='cuda'
                 ):
        self.device = device
        self.unet = unet
        self.added_linear = added_linear.to(device)
        self.config = config_file
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix',torch_dtype=torch.float16, use_auth_token=True,
            unet = unet,
            cache_dir='/mnt/store/jparanj1/.cache'
        ).to("cuda")

        self.model = load_model(config_file, grounded_checkpoint, device=device)
        # self.device = device
        # load model
        self.ram_model = ram(pretrained=ram_checkpoint,
                                            image_size=384,
                                            vit='swin_l')
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(device)
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.return_text = return_text
        self.return_boxes = return_boxes
        self.return_masks = return_masks
        self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-large-patch14',
            # subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            # subfolder="text_encoder",
        ).to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt = None,
        image = None,
        im_path = '',
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ["latents"],
        cross_attention_kwargs = None,
        **kwargs,
    ):
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Check inputs
        self.pipeline.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._image_guidance_scale = image_guidance_scale

        device = self.pipeline._execution_device

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        # 2. Encode input prompt
        prompt_embeds = self.pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.pipeline.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        #2.5 get gsam outputs
        text_embeddings, box_embeddings, mask_embeddings = get_gsam_outputs(
            im_path,
            model = self.model,
            ram_model = self.ram_model,
            sam_predictor = self.predictor,
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            box_threshold = self.box_threshold,
            text_threshold = self.text_threshold,
            iou_threshold = self.iou_threshold,
            return_boxes = self.return_boxes,
            return_masks = self.return_masks,
            return_text = self.return_text,
            device = self.device
        )

        #concatenate additional inputs from foundation model
        #legacy setting
        # fm_states = None
        # if text_embeddings is not None:
        #     fm_states = text_embeddings
        # if box_embeddings is not None:
        #     fm_states = torch.cat([fm_states, box_embeddings], dim=-1)
        # if mask_embeddings is not None:
        #     fm_states = torch.cat([fm_states, mask_embeddings], dim=-1)

        # fm_states = self.added_linear(fm_states.to(torch.float32).to(device))
        # fm_states = fm_states.unsqueeze(0).repeat(prompt_embeds.shape[0],1,1)
        # print(prompt_embeds.shape)
        # print(fm_states.shape)
        # encoder_hidden_states = torch.cat([prompt_embeds, fm_states], dim=1)
        
        #concatenate additional inputs from foundation model
        fm_states = None
        if mask_embeddings is not None:
            fm_states = mask_embeddings
        # elif box_embeddings is not None:
        #     fm_states = box_embeddings
        else:
            fm_states = box_embeddings
        # print("fm states only box or mask shape: ", fm_states.shape)
        # print("fm states only text shape: ", text_embeddings.shape)
        # print("encoder hidden states: ", prompt_embeds.shape)
        # if text_embeddings is not None:
        fm_states_text = (text_embeddings).unsqueeze(1).repeat(1,fm_states.shape[1],1)
        fm_states = torch.cat([fm_states, fm_states_text], dim=-1)

        fm_states = fm_states.unsqueeze(0).repeat(prompt_embeds.shape[0],1,1,1)
        
        # print("encoder hidden states shape: ", encoder_hidden_states.shape)
        # print("fm states shape: ", fm_states.shape)
        b, nb, sd, d = fm_states.shape
        fm_states = fm_states.reshape((b, nb*sd, d))
        # print("fm_states shape after combining: ", fm_states.shape)
        fm_states = self.added_linear(fm_states.to(prompt_embeds.dtype))
        encoder_hidden_states = torch.cat([prompt_embeds, fm_states], dim=1)
        
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.pipeline.do_classifier_free_guidance,
            )
        # 3. Preprocess image
        image = self.pipeline.image_processor.preprocess(image)

        # 4. set timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.pipeline.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.pipeline.do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.pipeline.vae_scale_factor
        width = width * self.pipeline.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.pipeline.vae.config.latent_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.pipeline.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.pipeline.unet.config} expects"
                f" {self.pipeline.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # 8.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        self.pipeline._num_timesteps = len(timesteps)
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 3) if self.pipeline.do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.pipeline.unet(
                    scaled_latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.pipeline.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.pipeline.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.pipeline.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.pipeline.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.pipeline.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.pipeline.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
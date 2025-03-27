import os
os.environ['HF_DATASETS_CACHE']="/mnt/store/jparanj1/.cache/"
os.environ['TRANSFORMERS_CACHE']='/mnt/store/jparanj1/.cache/'
os.environ['HF_HOME']="/mnt/store/jparanj1/.cache/"
os.environ['HF_HUB_CACHE']='/mnt/store/jparanj1/.cache/'
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2
from matplotlib import pyplot as plt
import torchvision.transforms.functional as tf
from copy import deepcopy
import torch
import torchvision
import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append("Grounded-Segment-Anything/GroundingDINO")
sys.path.append("Grounded-Segment-Anything/recognize-anything")

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
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
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

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)


class ThermalDataset(Dataset):
    def __init__(self, data_root, 
                 config_file='Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
                 grounded_checkpoint='Grounded-Segment-Anything/groundingdino_swint_ogc.pth', 
                 ram_checkpoint='Grounded-Segment-Anything/ram_swin_large_14m.pth',
                 sam_checkpoint='Grounded-Segment-Anything/sam_vit_b_01ec64.pth',
                 return_boxes=False,
                 return_masks=True,
                 return_text=True,
                 device='cuda'):
        
        self.data_root = data_root
        self.natural_im_list = sorted(os.listdir(os.path.join(data_root,"Vis")))
        self.thermal_im_list = sorted(os.listdir(os.path.join(data_root, "Ir")))
        self.natural_im_list = self.natural_im_list[:len(self.thermal_im_list)]
        self.p_enc_2d = PositionalEncodingPermute2D(256).to(device)

        with open('data_preparation/thermal_instructions.txt', 'r') as f:
            self.thermal_instructions = f.readlines()

        print("sample instructions: ", self.thermal_instructions[:5])
        print("Natural im list: ", len(self.natural_im_list))
        print("Thermal im list: ",len(self.thermal_im_list))

        self.column_names = ['visible_image', 'edit_instruction' ,'thermal_image', 'text_embed', 'box_embed', 'mask_embed']

        self.model = load_model(config_file, grounded_checkpoint, device=device)
        self.device = device
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
        self.return_text = True if return_text.lower()=='true' else False
        self.return_boxes = True if return_boxes.lower()=='true' else False
        self.return_masks = True if return_masks.lower()=='true' else False
        self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-large-patch14',
            # subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            # subfolder="text_encoder",
        ).to(self.device)
    def __len__(self):
        return len(self.natural_im_list)
    
    def __getitem__(self, i):
        # print(self.natural_im_list[i])
        natural_im = PIL.Image.open(os.path.join(self.data_root, 'Vis', self.natural_im_list[i]))
        thermal_im = PIL.Image.open(os.path.join(self.data_root, 'Ir', self.thermal_im_list[i]))
        text_ins = random.choice(self.thermal_instructions)

        #choose which wave to generate
        wave = 'long'
        if 'kaist' in self.data_root.lower():
            wave = 'long'
        elif 'flir' in self.data_root.lower():
            wave = 'long'
        elif 'osu' in self.data_root.lower():
            wave = 'mid'
        elif 'litiv' in self.data_root.lower():
            wave = 'mid'
        elif 'nirscene' in self.data_root.lower():
            wave = 'near'
        if text_ins[-1]!='.':
            text_ins = text_ins+'. '
        text_ins = text_ins + f'Make it {wave} wave Infrared.'

        #get GSAM outputs
        # print(f"return text: {self.return_text}, return boxes: {self.return_boxes}, return masks: {self.return_masks}")
        with torch.no_grad():
            image_path = os.path.join(self.data_root, 'Vis', self.natural_im_list[i])
            image_pil, image = load_image(image_path)
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
            raw_image  = transform(raw_image).unsqueeze(0).to(self.device)

            res = inference_ram(raw_image , self.ram_model)
            tags=res[0].replace(' |', ',')
            # print("predicted tags: ", tags)
            # run grounding dino model
            boxes_filt, scores, pred_phrases = get_grounding_output(
                self.model, image, tags, self.box_threshold, self.text_threshold, device=self.device
            )

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            # use NMS to handle overlapped boxes
            # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            # print(f"After NMS: {boxes_filt.shape[0]} boxes and total shape: {boxes_filt.shape}")
            # print("pred_phrases: ", pred_phrases)
            selected_tags = [p[:p.find('(')] for p in pred_phrases]
            # print("selected tags: ", selected_tags)
            if self.return_text:
                text_embeddings = self.text_encoder(tokenize_captions(self.tokenizer, selected_tags).to(self.device))[1]
                # print("text embeddings shape: ", text_embeddings.shape)
            else:
                tmp = self.text_encoder(tokenize_captions(self.tokenizer, selected_tags).to(self.device))[1]
                text_embeddings = torch.zeros_like(tmp).to(self.device)

            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
            if self.return_boxes:
                box_embeddings, _ = self.predictor.model.prompt_encoder(
                    points = None,
                    boxes = transformed_boxes,
                    masks = None
                )
                box_embeddings = box_embeddings.to(self.device)
                # box_embeddings = box_embeddings.to(self.device).reshape((boxes_filt.shape[0],-1))
                # print("box embeddings shape: ", box_embeddings.shape)
            else:
                if self.return_masks:
                    box_embeddings = None
                else:
                    tmp, _ = self.predictor.model.prompt_encoder(
                        points = None,
                        boxes = transformed_boxes,
                        masks = None
                    )
                    box_embeddings = torch.zeros_like(tmp).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            if self.return_masks:
                # masks, _, _ = self.predictor.predict_torch(
                #     point_coords = None,
                #     point_labels = None,
                #     boxes = transformed_boxes.to(self.device),
                #     multimask_output = False,
                # )
                # print("Mask shape: ", masks.shape)
                # print("image shape: ", image.shape)
                _, mask_embeddings = self.predictor.model.prompt_encoder(
                    points = None,
                    boxes = None,
                    masks = masks.float()
                )
                # print("mask embeddings shape: ", mask_embeddings.shape)
                mask_embeddings_with_pos_embed = mask_embeddings + (self.p_enc_2d(mask_embeddings))
                # print("mask embeddings with positional embeddings shape: ", mask_embeddings_with_pos_embed.shape)

                #average pool to 100 tokens
                b, c, h, w = mask_embeddings_with_pos_embed.shape
                mask_embeddings_with_pos_embed_avgpooled = torch.nn.functional.adaptive_avg_pool2d(mask_embeddings_with_pos_embed, (10,10))
                # print("mask embeddings with pos embed shape: ", mask_embeddings_with_pos_embed_avgpooled.shape)
                mask_embeddings = mask_embeddings_with_pos_embed_avgpooled.permute(0,2,3,1).reshape((b,100,c))
                # print("mask embeddings shape: ", mask_embeddings.shape)
                mask_embeddings = mask_embeddings.to(self.device)
            else:
                mask_embeddings = None
                    

            # # draw output image
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # for mask in masks:
            #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # for box, label in zip(boxes_filt, pred_phrases):
            #     show_box(box.numpy(), plt.gca(), label)

            # # plt.title('RAM-tags' + tags + '\n' + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
            # plt.axis('off')
            # plt.savefig(
            #     "automatic_label_output.jpg", 
            #     bbox_inches="tight", dpi=300, pad_inches=0.0
            # )

        # return natural_im, thermal_im, text_ins, text_embeddings, box_embeddings, mask_embeddings
        return {
            "visible_image": natural_im,
            "thermal_image": thermal_im,
            "edit_instruction": text_ins,
            "text_embed": text_embeddings,
            "box_embed": box_embeddings,
            "mask_embed": mask_embeddings,
            "masks": masks
        }


if __name__ == "__main__":
    # data_path = '/mnt/store/jparanj1/Thermal_Datasets/M3FD_Fusion/splits/split_1/train'
    data_path = '/mnt/store/jparanj1/Thermal_Datasets/OSU CT/train'
    d = ThermalDataset(data_path, return_boxes=True,
                 return_masks=True,
                 return_text=True,)
    d0 = d[0]
    print("Length of dataset: ", len(d))
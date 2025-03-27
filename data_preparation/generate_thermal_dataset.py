#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import os
os.environ['HF_DATASETS_CACHE']="/mnt/store/jparanj1/.cache/"
os.environ['TRANSFORMERS_CACHE']='/mnt/store/jparanj1/.cache/'
os.environ['HF_HOME']="/mnt/store/jparanj1/.cache/"
os.environ['HF_HUB_CACHE']='/mnt/store/jparanj1/.cache/'
os.environ['TFDS_DATA_DIR'] = '/mnt/store/jparanj1/'
import model_utils
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for InstructPix2Pix style training."
    )
    parser.add_argument("--dataset_id", type=str, default="osu")
    parser.add_argument("--max_num_samples", type=int, default=50000)
    parser.add_argument("--data_root", type=str, default="osu")
    parser.add_argument("--source_root", type=str, default='/mnt/store/jparanj1/Thermal_Datasets/OSU CT')
    args = parser.parse_args()
    return args

def main(args):
    print("Preparing the image pairs...")
    os.makedirs(args.data_root, exist_ok=True)
    for subfolder in os.listdir(args.source_root):
        if 'a' in subfolder:
            continue 
        if 'zip' in subfolder:
            continue
        if '4' in subfolder or '5' in subfolder:
            continue
        r1 = sorted(os.listdir(os.path.join(args.source_root,subfolder)))
        r2 = sorted(os.listdir(os.path.join(args.source_root,subfolder[0]+'a')))
        print(subfolder, len(r1), len(r2))

        for i, sample in enumerate(r1):
            try:
                thermal_sample = r2[i]
                original_image = Image.open(os.path.join(args.source_root,subfolder,sample)).convert("RGB")
                thermal_image = Image.open(os.path.join(args.source_root,subfolder[0]+'a',thermal_sample)).convert("RGB")

                sample_dir = os.path.join(args.data_root, subfolder+'_'+sample)
                os.makedirs(sample_dir)

                original_image.save(os.path.join(sample_dir, "original_image.png"))
                thermal_image.save(os.path.join(sample_dir, "thermal_image.png"))
            except:
                print(subfolder, sample)
                print("Fault here: Continuing")
                continue

    print(f"Total generated image-pairs: {len(os.listdir(args.data_root))}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

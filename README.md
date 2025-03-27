# Instruction-tuning Stable Diffusion
Code for the [paper: F-ViTA: Foundation Model Guided Visible to Thermal Translation](tbf)

## Table of contents

[Abstract](#abstract) <br>
[Data preparation](#data-preparation) <br>
[Checkpoint preparation](#checkpoint-preparation) <br>
[Training](#training) <br>
[Inference](#inference) <br>
[Acknowledgements](#acknowledgements) <br>

<p align="center">
<img src="resources/intro_fig.png"/>
</p>

## Abstract 

Thermal imaging is crucial for scene understanding, particularly in low-light and nighttime conditions. However, collecting large thermal datasets is costly and labor-intensive due to the specialized equipment required for infrared image capture. To address this challenge, researchers have explored visible-to-thermal image translation. Most existing methods rely on Generative Adversarial Networks (GANs) or Diffusion Models (DMs), treating the task as a style transfer problem. As a result, these approaches attempt to learn both the modality distribution shift and underlying physical principles from limited training data.  In this paper, we propose F-ViTA, a novel approach that leverages the general world knowledge embedded in foundation models to guide the diffusion process for improved translation. Specifically, we condition an InstructPix2Pix Diffusion Model with zero-shot masks and labels from foundation models such as SAM and Grounded DINO. This allows the model to learn meaningful correlations between scene objects and their thermal signatures in infrared imagery. Extensive experiments on five public datasets demonstrate that F-ViTA outperforms state-of-the-art (SOTA) methods. Furthermore, our model generalizes well to out-of-distribution (OOD) scenarios and can generate Long-Wave Infrared (LWIR), Mid-Wave Infrared (MWIR), and Near-Infrared (NIR) translations from the same visible image.

## Data preparation

For training on custom datasets, structure the data in the following format:
```
data_root
|---> train
      |---> Vis
            |---> img1.png
            |---> img2.png
            ...
      |---> Ir
            |---> img1.png
            |---> img2.png
            ...
|---> val
      |---> Vis
            |---> img1.png
            |---> img2.png
            ...
      |---> Ir
            |---> img1.png
            |---> img2.png
            ...
```
Here, Vis represents the visible image folder and Ir represents the corresponding thermal image folder

After this, add the dataset to the list of accepted datasets in finetune_instruct_pix2pix.py (line 855 onwards). Please follow the existing examples and add an additional conditional statement to add your dataset.

## Checkpoint Preparation
clone the Grounded SAM folder from IDEA-Research
```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```
Download these checkpoints and paste them in the Grounded-Segment-Anything folder

1) [RAM](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth)

2) [Grounded DINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

3) [SAM](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth)

Feel free to use other versions of these foundation models.

### F-ViTA Checkpoints:

Coming Soon
## Training

### Dev env setup


```
conda env create -f gsam.yml
conda activate gsam
```

### Launching training
Make necessary changes in the train_scrip.sh files including name of the output directory, dataset id and any other hyperparameters if required.

```
bash train_script.sh
```

## Inference

```
python inference_gsam.py <checkpoint-path> <save-name> <dataset-name>
```
An example is shown in the inference_gsam.sh


## Acknowledgements

Thanks to the amazing work by [Tim Brooks](https://github.com/timothybrooks/instruct-pix2pix) and [IDEA-Research](https://github.com/IDEA-Research/Grounded-Segment-Anything). Our work is  

## Citation

```bibtex
To be Added
```


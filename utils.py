import os
import sys
os.environ['HF_DATASETS_CACHE']="/mnt/store/jparanj1/.cache/"
os.environ['TRANSFORMERS_CACHE']='/mnt/store/jparanj1/.cache/'
os.environ['HF_HOME']="/mnt/store/jparanj1/.cache/"
os.environ['HF_HUB_CACHE']='/mnt/store/jparanj1/.cache/'

import shutil

def get_val_images(dataset_name):
    images = []
    gts = []
    names = []
    if dataset_name=='osu':
        root = '/mnt/store/jparanj1/Thermal_Datasets/OSU CT'      
        file_list = [f"img_{i:05}.bmp" for i in range(100000)]

        # Separate into odd and even lists
        odd_files = [file for i, file in enumerate(file_list) if i % 2 == 1]  # Odd index
        even_files = [file for i, file in enumerate(file_list) if i % 2 == 0]  # Even index

        for i in range(len(even_files)):
            if os.path.exists(os.path.join(root, '5b', even_files[i])) and os.path.exists(os.path.join(root, '5a', odd_files[i])):
                images.append(os.path.join(root, '5b', even_files[i]))
                gts.append(os.path.join(root, '5a', odd_files[i]))
                names.append('5b_'+ even_files[i])
                shutil.copy(gts[-1], 'osu_gts/'+names[-1])

        for i in range(len(even_files)):
            if os.path.exists(os.path.join(root, '6b', even_files[i])) and os.path.exists(os.path.join(root, '6a', odd_files[i])):
                images.append(os.path.join(root, '6b', even_files[i]))
                gts.append(os.path.join(root, '6a', odd_files[i]))
                names.append('6b_'+even_files[i])
                shutil.copy(gts[-1], 'osu_gts/'+names[-1])

    elif 'm3fd' in dataset_name:
        split = int(dataset_name[-1])
        root = f'/mnt/store/jparanj1/Thermal_Datasets/M3FD_Fusion/splits/split_{split}/val'
        file_list = os.listdir(os.path.join(root,'Vis'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'Vis', file_list[i]))
            gts.append(os.path.join(root, 'Ir', file_list[i]))
            names.append(file_list[i])

    elif 'flir' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/FLIR_Align/test'
        file_list = os.listdir(os.path.join(root,'Vis'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'Vis', file_list[i]))
            gts.append(os.path.join(root, 'Ir', file_list[i]))
            names.append(file_list[i])

    elif 'kaist' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/KAIST/test'
        file_list = os.listdir(os.path.join(root,'Vis'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'Vis', file_list[i]))
            gts.append(os.path.join(root, 'Ir', file_list[i]))
            names.append(file_list[i])

    elif 'llvip' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/LLVIP/test'
        file_list = os.listdir(os.path.join(root,'Vis'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'Vis', file_list[i]))
            gts.append(os.path.join(root, 'Ir', file_list[i]))
            names.append(file_list[i])
    
    # elif 'litiv' in dataset_name:
    #     root = f'/mnt/store/jparanj1/Thermal_Datasets/litiv2012_dataset_2/test'
    #     file_list = os.listdir(os.path.join(root,'Vis'))
    #     for i in range(len(file_list)):
    #         images.append(os.path.join(root, 'Vis', file_list[i]))
    #         gts.append(os.path.join(root, 'Ir', file_list[i]))
    #         names.append(file_list[i])
    elif 'litiv' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/litiv2012_dataset/SEQUENCE7'
        file_list = os.listdir(os.path.join(root,'VISIBLE/input'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'VISIBLE/input', file_list[i]))
            gts.append(os.path.join(root, 'THERMAL/input', file_list[i]))
            names.append(file_list[i])

    elif 'mfnet' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/MFNet'
        file_list = os.listdir(os.path.join(root,'RGB'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'RGB', file_list[i]))
            gts.append(os.path.join(root, 'Modal', file_list[i]))
            names.append(file_list[i])
        
    elif 'nirscene' in dataset_name:
        root = f'/mnt/store/jparanj1/Thermal_Datasets/NIRSCENE/test'
        file_list = os.listdir(os.path.join(root,'Vis'))
        for i in range(len(file_list)):
            images.append(os.path.join(root, 'Vis', file_list[i]))
            gts.append(os.path.join(root, 'Ir', file_list[i]))
            names.append(file_list[i])
    else:
        print("Assuming data directory path in dataset_name")
        file_list = os.listdir(dataset_name)
        for i in range(len(file_list)):
            images.append(os.path.join(dataset_name, file_list[i]))
            gts.append(os.path.join(dataset_name, file_list[i]))
            names.append(file_list[i])

    print("len images: ", len(images))
    print("len gts: ", len(gts))
    return images, gts, names

from cleanfid import fid
from pathlib import Path
from cleanfid import fid
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import lpips
from glob import glob
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes


class PairDataset(Dataset):
    def __init__(self, dataset1, dataset2) -> None:
        self.dataset1 = Path(dataset1)

        self.dataset1 = glob(str(Path(self.dataset1) / "**" / "*.*"), recursive=True)
        self.dataset1 = sorted(
            x.replace("/", os.sep)
            for x in self.dataset1
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        if dataset2 == 'osu':
            _, self.dataset2, _ = get_val_images('osu')
        else:
            self.dataset2 = Path(dataset2)
            self.dataset2 = glob(str(Path(self.dataset2) / "**" / "*.*"), recursive=True)
        self.dataset2 = sorted(
            x.replace("/", os.sep)
            for x in self.dataset2
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)
        assert self.len1 == self.len2, "unpaired datasets"
        self.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def __len__(self):
        return self.len1

    def __getitem__(self, index):
        img1_path = self.dataset1[index]
        img2_path = self.dataset2[index]

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img1 = img1.convert("L")
        img2 = img2.convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def get_lpips_ssim_metrics(dataset1, dataset2, log_file, batch_size=64):
    dataset = PairDataset(dataset1, dataset2)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size
    )

    lpips_metric = lpips.LPIPS(net="alex")
    ddim_metric = lpips.DSSIM(colorspace="RGB")
    total_num = len(dataset)

    init_lpips = 0
    init_ssim = 0
    init_psnr = 0

    with torch.no_grad():
        for i, (img1, img2) in tqdm(
            enumerate(dataloader),
            desc="LPIPS",
            initial=0,
            total=int(total_num / batch_size),
        ):
            img1, img2 = img1.cuda(), img2.cuda()
            lpips_metric = lpips_metric.cuda()
            d_lpips = lpips_metric.forward(img1, img2)
            init_lpips = init_lpips + d_lpips.sum()

    final_lpips = init_lpips / (total_num + 1e-8)
    with open(log_file, "a") as f:
        f.write("lpips:{0}".format(final_lpips))

    dataloader2 = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=1
    )
    with torch.no_grad():
        for i, (img1, img2) in tqdm(
            enumerate(dataloader2), desc="SSIM+PSNR", initial=0, total=total_num
        ):
            img1, img2 = img1.cuda(), img2.cuda()
            ddim_metric = ddim_metric.cuda()
            d_ssim = ddim_metric.forward(img1, img2)
            init_ssim = init_ssim + d_ssim

            d_psnr = psnr(img1, img2)
            init_psnr = init_psnr + d_psnr

    final_dssim = init_ssim / (total_num + 1e-8)
    final_psnr = init_psnr / (total_num + 1e-8)

    with open(log_file, "a") as f:
        f.write(
            "dssim:{0},ssim:{1},psnr:{2}\n".format(
                final_dssim, 1.0 - 2.0 * (final_dssim), final_psnr
            )
        )


def write_log(
    dataset1,
    dataset2,
    log_file,
    fid_score_pytorch_v3,
    fid_score_clean_v3,
    fid_score_clean_clip,
    kid_score_pytorch_v3,
):
    with open(log_file, "a") as f:
        f.write("first_dataset:{0},second_daatset:{1}\n".format(dataset1, dataset2))
        f.write(
            "fid_score_pytorch_v3:{0},fid_score_clean_v3:{1},fid_score_clean_clip:{2},kid_score_pytorch_v3:{3}\n".format(
                fid_score_pytorch_v3,
                fid_score_clean_v3,
                fid_score_clean_clip,
                kid_score_pytorch_v3,
            )
        )


def get_all_fid(dataset1, dataset2, log_file):
    if dataset2=='osu':
        _, dataset2, _ = get_val_images('osu')
    fid_score_clean_v3 = fid.compute_fid(
        dataset1,
        dataset2,
        mode="clean",
        model_name="inception_v3",
        num_workers=64,
        batch_size=64,
    )
    fid_score_pytorch_v3 = fid.compute_fid(
        dataset1,
        dataset2,
        mode="legacy_pytorch",
        model_name="inception_v3",
        num_workers=64,
        batch_size=64,
    )
    fid_score_clean_clip = fid.compute_fid(
        dataset1,
        dataset2,
        mode="clean",
        model_name="clip_vit_b_32",
        num_workers=64,
        batch_size=64,
    )
    kid_score_pytorch_v3 = fid.compute_kid(
        dataset1, dataset2, mode="clean", num_workers=64, batch_size=64
    )
    write_log(
        dataset1,
        dataset2,
        log_file,
        fid_score_pytorch_v3,
        fid_score_clean_v3,
        fid_score_clean_clip,
        kid_score_pytorch_v3,
    )


if __name__ == "__main__":

    #one time only to make gts from osu val
    # _,_,_ = get_val_images('osu')

    # log_file = "/mnt/store/jparanj1/instruction-tuned-sd/metrics/metric_fid_nirscene-finetuned-gsam-masks-checkpoint-3000-100steps_on_nirscene"
    log_file = "/mnt/store/jparanj1/instruction-tuned-sd/metrics/osu_finetuned_ablation_only_boxes_checkpoint-3000_results"
    real_path = "/mnt/store/jparanj1/Thermal_Datasets/OSU CT/val/Ir"
    generated_path = "/mnt/store/jparanj1/instruction-tuned-sd/predictions/osu_finetuned_ablation_only_boxes_checkpoint-3000"
    get_all_fid(real_path, generated_path, log_file)
    get_lpips_ssim_metrics(real_path, generated_path, log_file)

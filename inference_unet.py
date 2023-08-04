import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm


channel_means = [0.485, 0.456, 0.406]
channel_stds  = [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])


def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)) # .cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    # mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    mask = cv.resize(mask, (img.shape[1], img.shape[0]), cv.INTER_AREA)
    return mask


def evaluate_img_via_subimages(model, img, subimage_size):
    if subimage_size <= 0:
        return evaluate_img(model, img)
    from preprocess_image import cut_image_into_subimages, join_subimages
    subimages = cut_image_into_subimages(img, subimage_size)
    # print("Shape of subimages:", [s.shape for _, _, s in subimages])
    masks = [(x, y, evaluate_img(model, subimage)) for x, y, subimage in tqdm(subimages)]
    # print("Shape of masks:", [m.shape for _, _, m in masks])
    mask = join_subimages(masks, (img.shape[0], img.shape[1]))
    return mask


def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)) # .cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])


def main(args_out_viz_dir, args_out_pred_dir, args_model_type, args_model_path, args_img_dir, args_subimage_size, args_threshold):
    if args_out_viz_dir != '':
        os.makedirs(args_out_viz_dir, exist_ok=True)
        for path in Path(args_out_viz_dir).glob('*.*'):
            os.remove(str(path))

    if args_out_pred_dir != '':
        os.makedirs(args_out_pred_dir, exist_ok=True)
        for path in Path(args_out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args_model_type == 'vgg16':
        model = load_unet_vgg16(args_model_path)
    elif args_model_type  == 'resnet101':
        model = load_unet_resnet_101(args_model_path)
    elif args_model_type  == 'resnet34':
        model = load_unet_resnet_34(args_model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()

    
    paths = [path for path in Path(args_img_dir).glob('*.*')]
    for path in tqdm(paths):
        # print(str(path))
        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        # print("Shape img0:", img_0.shape)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        # img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img_via_subimages(model, img_0, args_subimage_size)  # evaluate_img(model, img_0)
        # ce zelimo kaj ven dobiti: yield path, img_0, prob_map_full

        if args_out_pred_dir != '':
            cv.imwrite(filename=join(args_out_pred_dir, f'{path.stem}.jpg'), img=(prob_map_full * 255).astype(np.uint8))

        if args_out_viz_dir != '':
            # plt.subplot(121)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            else:
                img_1 = img_0

            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()

            prob_map_patch = evaluate_img_patch(model, img_1)

            # plt.title(f'name={path.stem}. \n cut-off threshold = {args_threshold}', fontsize=4)
            prob_map_viz_patch = prob_map_patch.copy()
            prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
            prob_map_viz_patch[prob_map_viz_patch < args_threshold] = 0.0
            fig = plt.figure()
            st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args_threshold}', fontsize="x-large")
            ax = fig.add_subplot(231)
            ax.imshow(img_1)
            ax = fig.add_subplot(232)
            ax.imshow(prob_map_viz_patch)
            ax = fig.add_subplot(233)
            ax.imshow(img_1)
            ax.imshow(prob_map_viz_patch, alpha=0.4)

            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full[prob_map_viz_full < args_threshold] = 0.0

            ax = fig.add_subplot(234)
            ax.imshow(img_0)
            ax = fig.add_subplot(235)
            ax.imshow(prob_map_viz_full)
            ax = fig.add_subplot(236)
            ax.imshow(img_0)
            ax.imshow(prob_map_viz_full, alpha=0.4)

            plt.savefig(join(args_out_viz_dir, f'{path.stem}.jpg'), dpi=500)
            plt.close('all')

        gc.collect()
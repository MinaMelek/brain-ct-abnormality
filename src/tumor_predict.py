# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:34:58 2022

@author: Mina
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import densenet121
from preprocessing import load_batch, standardize
import argparse


# define pytorch arch
class densenet121_tumor(nn.Module):
    def __init__(self, pretrained=False, class_num=2, seed=0):
        """
        Define the tumor model architecture.
        
        :param pretrained: a boolean value for whether to use pretrained weights or not, default=False.
        :param class_num: an integer representing the number of classes, default=2.
        :param seed: an integer representing the random state of weights initialization, default=0.
        """
        super(densenet121_tumor, self).__init__()
        self.class_num = class_num
        self.seed = torch.manual_seed(seed)
        self.densenet121 = densenet121(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(1024, 64)
        self.norm = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, class_num if class_num > 2 else 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(class_num)

    def forward(self, x):
        x = self.densenet121(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.dense(x)
        # x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x) if self.class_num <= 2 else self.softmax(x)

        return x


def load_model(model_path, gpu=True, parallel=False, gpu_index=None):
    # TODO: structure the same model for stroke and use the same script for both
    if gpu_index:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    model = densenet121_tumor(pretrained=True, class_num=2)
    state = torch.load(model_path)
    model.load_state_dict(state)

    model.eval()
    if gpu:
        model = model.cuda()
    if parallel:
        model = nn.DataParallel(model)

    return model


def predict(input_batch: torch.float, model, gpu=True, log=True):
    # prediction process
    if gpu:
        input_batch = input_batch.cuda()
    prediction = model(torch.autograd.Variable(input_batch.float())).view(-1)
    confs = prediction.detach().cpu().numpy() if gpu else prediction.detach().numpy()
    if log:
        for i, conf in enumerate(confs):
            print(f"slice_{i}: " + "{} with confidence {:.2f}%"
                  .format(*('Tumor', conf * 100) if conf > 0.5 else ('Normal', (1 - conf) * 100)))

    return confs


def main(im=None, image_path='', model_path=None, mode=None, gpu=True, parallel=False, gpu_index=None):
    """
    Apply tumor model

    :param im: an image array (or a list of images if mode='batch') with size=256x256x3, default=None.
    :param image_path: path to brain image (or a directory of images) in case of im=None, default=''.
    :param model_path: path to pytorch model state, default='models/JUH_noisy_model.pt'
    :param mode: a string value represent the input mode whether to be 'single'; an image or a path to an image,
                 or 'batch'; an array of images or a path to a directory, default=None.
    :param gpu: a bool indicator to whether using gpu or not, default=True.
    :param parallel: a bool indicator to whether using multiple devices in parallel or not, default=False.
    :param gpu_index: the used gpu devices indices (can use multiple devices if parallel=True),
                      for example; gpu_index='0' lets you use device:0, so as gpu_index='1,2', default='0'(None).
    :return: a numerical value representing the prediction confidence interval.
    """
    # Read image
    if mode == 'batch':
        assert os.path.isdir(image_path) or len(im.shape) == 4, "selecting batch-mode, yet a single file is passed."
        im = load_batch(list(Path(image_path).glob('*.png'))) if im is None else im
        im = torch.from_numpy(im)  # Convert to tensor
    elif mode == 'single':
        assert os.path.isfile(image_path) or len(im.shape) == 3, "please, assign argument --mode batch"
        im = load_batch([image_path]) if im is None \
            else standardize(im).reshape(1, *im.shape[-3:])  # prepare raw image
        im = torch.from_numpy(im)  # Convert to tensor
    else:
        pass  # mode is None in case of called from main ## deprecated

    # Load model
    if model_path is None:
        model_path = '../models/JUH_noisy_model.pt'
    model = load_model(model_path, gpu, parallel, gpu_index)

    # Predict
    confs = predict(im, model)
    return confs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help='load and predict on gpu, default: False (cpu)')
    parser.add_argument('--gpu_index', dest='gpu_index', type=str, default='0',
                        help='gpu index if you have multiple gpus, default: 0')
    parser.add_argument('--parallel', dest='parallel', default=False, action='store_true',
                        help='use model in parallel mode in case of multiple devices, default: False')
    parser.add_argument('--mode', dest='mode', type=str, default='single',
                        help='select whether to get prediction on a single image or batch, default: single')
    parser.add_argument('-m-pth', '--model_path', dest='model_path', type=str,
                        default=os.path.join('..', 'models', 'JUH_noisy_model.pt'),
                        help='select pytorch model path, default: "../models/JUH_noisy_model.pt"')
    parser.add_argument('-i-pth', '--image_path', dest='image_path', type=str, required=True,
                        help='path of image to produce prediction. (required)')
    parser.add_argument('-b', dest='benchmark_mode', default=False, action='store_true',
                        help='indicate a benchmarking mode, default: False')

    args = parser.parse_args()

    # For benchmarking
    if args.benchmark_mode and torch.cuda.is_available():
        import nvidia_dlprof_pytorch_nvtx

        nvidia_dlprof_pytorch_nvtx.init()
        torch.backends.cudnn.benchmark = True

    _ = predict(image_path=args.image_path, model_path=args.model_path,
                mode=args.mode, gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)



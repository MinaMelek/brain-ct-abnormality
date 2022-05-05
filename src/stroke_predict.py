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

# classes definitions
stroke_classes = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 
                  'Epidural', 'Subdural', 'No_Hemorrhage', 'Fracture_Yes_No']
class_num = len(stroke_classes)


# define pytorch arch
class densenet121_stroke(nn.Module):
    def __init__(self, pretrained=False, seed=0):
        """
        Define the stroke model architecture.

        :param pretrained: a boolean value for whether to use pretrained weights or not, default=False.
        :param seed: an integer representing the random state of weights initialization, default=0.
        """
        super(densenet121_stroke, self).__init__()
        self.class_num = class_num
        self.seed = torch.manual_seed(seed)
        self.densenet121 = densenet121(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, class_num if class_num > 2 else 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)

        return x


def load_model(model_path, gpu, parallel, gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    model = densenet121_stroke(pretrained=True)
    if gpu:
        model = model.cuda()
    if parallel:
        model = nn.DataParallel(model)
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])

    return model


def stroke_predict(im=None, image_path='', model_path=None, mode='batch', gpu=True, parallel=True, gpu_index='0'):
    """
    Apply hemorrhage and fracture model

    :param im: an image array (or a list of images if mode='batch') with size=256x256x3, default=None.
    :param image_path: path to brain image (or a directory of images) in case of im=None, default=''.
    :param model_path: path to pytorch model state, default='models/CTish_frac_model.pt'
    :param mode: a string value represent the input mode whether to be 'single'; an image or a path to an image,
                 or 'batch'; an array of images or a path to a directory, default='batch'.
    :param gpu: a bool indicator to whether using gpu or not, default=True.
    :param parallel: a bool indicator to whether using multiple devices in parallel or not, default=False.
    :param gpu_index: the used gpu devices indices (can use multiple devices if parallel=True), default='0'.
    :return: a numerical value representing the prediction confidence interval.
    """
    # Load model
    if model_path is None:
        model_path = '../models/CTish_frac_model.pt'
    model = load_model(model_path, gpu, parallel, gpu_index)
    model.eval()
    # Read image
    if mode == 'batch':
        assert os.path.isdir(image_path) or len(im.shape) == 4, "selecting batch-mode, yet a single file is passed."
        im_norm = load_batch(list(Path(image_path).glob('*.png'))) if im is None else im
    else:  # single
        assert os.path.isfile(image_path) or len(im.shape) == 3, "please, assign argument --mode batch"
        im_norm = load_batch([image_path]) if im is None \
            else standardize(im).reshape(1, *im.shape[-3:])  # prepare raw image
    im_norm = torch.FloatTensor(im_norm)  # Convert to tensor
    if gpu:
        im_norm = im_norm.cuda()
    # Predict
    predict = model(torch.autograd.Variable(im_norm)).sigmoid()#.argmax(1)
    confs = predict.detach().cpu().numpy() if gpu else predict.detach().numpy()
    for i, conf in enumerate(confs):
        cls_id = conf.argmax()
        print(f"slice_{i}: \"{stroke_classes[cls_id]}\" with confidence {conf[cls_id]*100:.2f}%")
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
                        default=os.path.join('..', 'models', 'CTish_frac_model.pt'),
                        help='select pytorch model path, default: "../models/CTish_frac_model.pt"')
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
    
    _ = stroke_predict(image_path=args.image_path, model_path=args.model_path,
                       mode=args.mode, gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:34:58 2022

@author: Mina
"""

import os
import torch
import torch.nn as nn
from torchvision.models import densenet121
import cv2
import argparse


# define pytorch arch
class densenet121_change_avg(nn.Module):
    def __init__(self, pretrained=False, class_num=2, seed=0):
        """
        Define the tumor model architecture.
        
        :param pretrained: a boolean value for whether to use pretrained weights or not, default=False.
        :param class_num: an integer representing the number of classes, default=2.
        :param seed: an integer representing the random state of weights initialization, default=0.
        """
        super(densenet121_change_avg, self).__init__()
        self.class_num = class_num
        self.seed = torch.manual_seed(seed)
        self.densenet121 = densenet121(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(1024, 64)
        self.norm = nn.BatchNorm1d(64, momentum=0.95, eps=0.005)
        self.output = nn.Linear(64, class_num if class_num > 2 else 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(class_num)

    def forward(self, x):
        x = self.densenet121(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.dense(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x) if self.class_num <= 2 else self.softmax(x)

        return x


def load_model(model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    model = densenet121_change_avg(pretrained=True, class_num=2)
    state = torch.load(model_path)
    model.load_state_dict(state)
    if args.gpu:
        model = model.cuda()
    if args.parallel:
        model = nn.DataParallel(model)
    return model


def tumor_predict(im=None, image_path='', model_path='models/JUH_noisy_model.pt'):
    """
    Apply tumor model

    :param im: an image array with size=512x512, default=None.
    :param image_path: path to brain image in case of im=None, default=''.
    :param model_path: path to pytorch model state, default='models/JUH_noisy_model.pt'
    :return: a numerical value representing the prediction confidence interval.
    """
    # Load model
    model = load_model(model_path)
    model.eval()
    # Read image
    im = cv2.imread(image_path) if im is None else im
    # Get image into shape
    im_norm = (im - im.mean()) / im.std()  # Standardize image
    im_norm = im_norm.transpose(2, 0, 1)  # Move channel first
    im_norm = torch.FloatTensor(im_norm.reshape(1, *im_norm.shape))  # Convert to tensor
    if args.gpu:
        im_norm = im_norm.cuda()
    # Predict
    predict = model(torch.autograd.Variable(im_norm)).view(-1)
    conf = predict.detach().cpu().numpy()[0] if args.gpu else predict.detach().numpy()[0]
    print("{} with confidence {:.2f}%".format(*('Tumor', conf * 100) if conf > 0.5 else ('Normal', (1 - conf) * 100)))
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help='load and predict on gpu, default: False (cpu)')
    parser.add_argument('--gpu_index', dest='gpu_index', type=str, default='0',
                        help='gpu index if you have multiple gpus, default: 0')
    parser.add_argument('--parallel', dest='parallel', default=False, action='store_true',
                        help='use model in parallel mode in case of multiple devices, default: False')
    parser.add_argument('-m-pth', '--model_path', dest='model_path', type=str,
                        default=os.path.join('models', 'JUH_noisy_model.pt'),
                        help='select pytorch model path, default: "models/JUH_noisy_model.pt"')
    parser.add_argument('-i-pth', '--image_path', dest='image_path', type=str, required=True,
                        help='path of image to produce prediction. (required)')

    args = parser.parse_args()
    _ = tumor_predict(image_path=args.image_path, model_path=args.model_path)

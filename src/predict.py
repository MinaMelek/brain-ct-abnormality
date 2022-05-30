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
class densenet121_multi(nn.Module):
    def __init__(self, pretrained=False, class_num=2, seed=0):
        """
        Define a general model architecture, for the different models.

        :param pretrained: a boolean value for whether to use pretrained weights or not, default=False.
        :param class_num: an integer representing the number of classes, default=2.
        :param seed: an integer representing the random state of weights initialization, default=0.
        """
        super(densenet121_multi, self).__init__()
        self.class_num = class_num
        self.seed = torch.manual_seed(seed)
        self.densenet121 = densenet121(pretrained=pretrained).features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(1024, class_num if class_num > 2 else 1)

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = self.output(x)

        return x


class model_eval(object):
    def __init__(self, model_type, gpu=True, parallel=False, gpu_index=None):
        """
        Define models and prediction parameters
        :param model_type: a string type of problem the model predicts, whether it's Hemorrhage, Tumor or Abnormality.
        :param gpu: a bool indicator to whether using gpu or not, default=True.
        :param parallel: a bool indicator to whether using multiple devices in parallel or not, default=False.
        :param gpu_index: the used gpu devices indices (can use multiple devices if parallel=True), for example;
                          gpu_index='0' lets you use device:0, so as gpu_index='1,2' (use with parallel), default=None.
        """
        self.model_type = model_type.lower()
        self.gpu = gpu
        self.parallel = parallel
        self.gpu_index = gpu_index
        self.model = None
        if self.model_type == 'hemorrhage':
            # classes definitions
            self.classes = ['Hemorrhage_exists', 'Epidural', 'Intraparenchymal',
                            'Intraventricular', 'Subarachnoid', 'Subdural']
        elif self.model_type == 'abnormal':
            self.classes = ['Fracture', 'CalvarialFracture', 'OtherFracture',
                            'MassEffect', 'MidlineShift']
        elif self.model_type == 'tumor':
            self.classes = ['Tumor', 'Normal']
        else:
            raise ValueError("model_type should be 'hemorrhage', 'tumor' or 'abnormal'")
        self.class_num = len(self.classes)

    def load_model(self, model_path):
        """
        Load model from disk (separate from predict function)
        :param model_path: path to pytorch model state.
        :return: the model object.
        """
        if self.gpu_index:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_index
        model = densenet121_multi(pretrained=True, class_num=self.class_num)
        if self.gpu:
            model = model.cuda()

        if self.model_type == 'hemorrhage':  # TODO: review
            # It is mandatory to use parallelism in case of the hemorrhage model,
            # as it was used in training due to the large training dataset.
            model = nn.DataParallel(model)
        state = torch.load(model_path)
        model.load_state_dict(state)

        if self.parallel and self.model_type != 'hemorrhage':
            model = nn.DataParallel(model)

        self.model = model
        return model

    def predict(self, input_batch: torch.float):
        """
        Prediction process
        :param input_batch: a batch of records (image_slices)
        :return: confidence of each record in the input batch
        """
        self.model.eval()

        if self.gpu:
            input_batch = input_batch.cuda()
        prediction = self.model(torch.autograd.Variable(input_batch.float())).sigmoid()
        if self.class_num <= 2:
            prediction = prediction.view(-1)
        confs = prediction.detach().cpu().numpy() if self.gpu else prediction.detach().numpy()

        return confs


def main(im=None, image_path='', model_dir=None, mode=None, gpu=True, parallel=False, gpu_index=None, log=True):
    """
    Apply models

    :param im: an image array (or a list of images if mode='batch') with size=256x256x3, default=None.
    :param image_path: path to brain image (or a directory of images) in case of im=None, default=''.
    :param model_dir: path to pytorch models directory, default='./models/'
    :param mode: a string value represent the input mode whether to be 'single'; an image or a path to an image,
                 or 'batch'; an array of images or a path to a directory, default=None.
    :param gpu: a bool indicator to whether using gpu or not, default=True.
    :param parallel: a bool indicator to whether using multiple devices in parallel or not, default=False.
    :param gpu_index: the used gpu devices indices (can use multiple devices if parallel=True), for example;
                      gpu_index='0' lets you use device:0, so as gpu_index='1,2' (use with parallel), default=None('0').
    :param log: a bool indicator for printing prediction result, default=True.
    # :return: a numerical value representing the prediction confidence interval. (deprecated)
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

    if model_dir is None:
        model_dir = Path('../models/final')
    # 1. Hemorrhage model:
    eval_hem = model_eval('hemorrhage', gpu, parallel, gpu_index)
    # Load model
    model_path = model_dir / 'Hemorrhage.pt'
    eval_hem.load_model(model_path)
    # Predict
    confs = eval_hem.predict(im)
    # Print results
    if log:
        for i, conf in enumerate(confs):
            cls_id = conf.argmax()
            print(f"slice_{i}: \"{eval_hem.classes[cls_id]}\" with confidence {conf[cls_id]*100:.2f}%")

    # 2. Abnormal model:
    eval_abn = model_eval('abnormal', gpu, parallel, gpu_index)
    # Load model
    model_path = model_dir / 'Abnormal.pt'
    eval_abn.load_model(model_path)
    # Predict
    confs = eval_abn.predict(im)
    # Print results
    if log:
        for i, conf in enumerate(confs):
            cls_id = conf.argmax()
            print(f"slice_{i}: \"{eval_abn.classes[cls_id]}\" with confidence {conf[cls_id]*100:.2f}%")

    # 3. Tumor model:
    eval_tmr = model_eval('tumor', gpu, parallel, gpu_index)
    # Load model
    model_path = model_dir / 'Tumor.pt'
    eval_tmr.load_model(model_path)
    # Predict
    confs = eval_tmr.predict(im)
    # Print results
    if log:
        for i, conf in enumerate(confs):
            print(f"slice_{i}: " + "{} with confidence {:.2f}%"
                  .format(*('Tumor', conf * 100) if conf > 0.5 else ('Normal', (1 - conf) * 100)))


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
    parser.add_argument('-m-pth', '--model_dir', dest='model_dir', type=str,
                        default=os.path.join('..', 'models'),
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

    main(image_path=args.image_path, model_dir=args.model_dir,
         mode=args.mode, gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)



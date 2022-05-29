import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocessing import slice_preprocess, download_data, standardize
from tumor_predict import load_model, predict as tumor_predict
from stroke_predict import predict as stroke_predict, stroke_classes
import json
import argparse


class Logger(object):
    def __init__(self, log_file="../log.txt"):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def __del__(self):
        self.log.close()
        self.log = self.terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class ImageDataset(Dataset):
    def __init__(self, target_files, dir_path, size, transform=None, stack=False):
        self.target_files = target_files
        self.dir_path = dir_path
        self.size = size
        self.transform = transform
        self.stack = stack

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        img_path = self.dir_path / self.target_files[idx]
        # Prepare image (slice) for prediction
        image = slice_preprocess(img_path, self.size)
        # stack mode means combine 3 slices
        if self.stack:
            if idx > 0:
                img_path = self.dir_path / self.target_files[idx-1]
                image_pre = slice_preprocess(img_path, self.size)
            else:
                image_pre = image.copy()

            if idx < self.__len__()-1:
                img_path = self.dir_path / self.target_files[idx+1]
                image_post = slice_preprocess(img_path, self.size)
            else:
                image_post = image.copy()
        else:
            image_pre = image.copy()
            image_post = image.copy()
        # Convert to rgb image
        image = np.concatenate([image_pre[:, :, np.newaxis], image[:, :, np.newaxis], image_post[:, :, np.newaxis]], 2)
        if self.transform:
            image = self.transform(image)
        return image


def main():
    patient_dir = download_data(args.url)
    patient_id = Path(patient_dir).name
    i = 0
    series = {}
    result = {}
    for dir_path, dir_names, filenames in os.walk(patient_dir):
        # Filter to Dicom image only.
        filenames = [fi for fi in filenames if fi.endswith(".dcm")]
        # Ignore paths with no dicom files.
        if len(filenames) == 0 or len(dir_names) > 0:
            continue
        # Determine patient-id/name, study-id and series-id.
        dir_path = Path(dir_path)
        # TODO: Decide how to use these ids
        study_id, series_id = dir_path.parts[-2:]
        print(f"{dir_path}: Study: {study_id}, Series: {series_id}")
        if study_id not in result.keys():
            result[study_id] = []
        # Only middle slices should be selected
        n_files = len(filenames)
        ignored_portion = n_files // 5
        target_files = filenames[ignored_portion: -ignored_portion]
        # TODO: redesign for efficiency
        new_size = (args.image_size, args.image_size)

        # PREDICTION
        batch_size = 64
        test_data = ImageDataset(target_files, dir_path, new_size, transform=standardize)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=24)
        # preload to remove overhead
        tumor_model = load_model(os.path.join(args.model_dir, 'JUH_noisy_model.pt'),
                                 gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)
        predict_1 = np.zeros(shape=(len(test_data), len(stroke_classes)))
        predict_2 = np.zeros(shape=(len(test_data),))
        for j, images in enumerate(test_dataloader):
            p_1 = stroke_predict(images, model_path=os.path.join(args.model_dir, 'CTish_frac_model.pt'),
                                 gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)
            p_2 = tumor_predict(images, tumor_model, gpu=args.gpu)  # TODO: remake for stroke, use args for inputs
            idx = j*batch_size
            predict_1[idx: idx+len(images)] = p_1
            predict_2[idx: idx+len(images)] = p_2
        series[i] = [dir_path, predict_1, predict_2]
        # format stroke prediction
        # out_1 = dict(zip(stroke_classes, map(lambda x: f"{x * 100:.2f}%", predict_1.max(0))))  # Todo: Change mean
        out_1 = predict_1.max(0)
        # format tumor prediction
        # tumor_mean = (predict_2 > 0.5).mean()  # percentage of tumor slices
        from collections import defaultdict
        oc, h = defaultdict(lambda x=0: x), 0
        for itm in predict_2:
            if itm > 0.5:
                oc[h] += 1
            else:
                h += 1
        max_occ = max(oc.values())  # max number of consecutive slices predicted as tumor
        # out_2 = {'Tumor': f"{predict_2.max() * 100:.2f}%" if max_occ > 2 else
        #                   f"{predict_2.mean() * 100:.2f}%"}
        out_2 = predict_2[predict_2 > 0.5].mean() if max_occ > 2 else predict_2[predict_2 <= 0.5].mean()
        result[study_id].append([*out_1, out_2])

        i += 1

    study = {}
    for key, val in result.items():
        study_out = np.array(val).mean(0)
        study[key] = {'Hemorrhage': dict(zip(stroke_classes, study_out[:-1])),
                      'Tumor': study_out[-1]}

    return series, {patient_id: study}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gpu', dest='gpu', default=False, action='store_true',
                        help='load and predict on gpu, default: False (cpu)')
    parser.add_argument('-index', dest='gpu_index', type=str, default=None,
                        help='gpu index if you have multiple gpus, default: None')
    parser.add_argument('-parallel', dest='parallel', default=False, action='store_true',
                        help='use model in parallel mode in case of multiple devices, default: False')
    parser.add_argument('-m-dir', dest='model_dir', type=str, default=os.path.join('..', 'models'),
                        help='select pytorch models directory, default: "../models/"')
    parser.add_argument('-o', dest='output_file', type=str, default=os.path.join('..', 'output', 'output.json'),
                        help='path for output json file, default: "../output/output.json"')
    parser.add_argument('-url', dest='url', type=str, required=True,
                        help='url for patient scans to produce prediction. (required)')
    parser.add_argument('-size', dest='image_size', type=int, default=256,
                        help='input images size, default: 256')
    parser.add_argument('-b', dest='benchmark_mode', default=False, action='store_true',
                        help='indicate a benchmarking mode, default: False')

    args = parser.parse_args()

    # For benchmarking
    if args.benchmark_mode:
        import nvidia_dlprof_pytorch_nvtx
        import torch
        nvidia_dlprof_pytorch_nvtx.init()
        torch.backends.cudnn.benchmark = True
        torch.autograd.profiler.emit_nvtx()

    default_stdout = sys.stdout
    sys.stdout = Logger()
    assert os.path.splitext(args.output_file)[-1] == '.json', 'Output file extension is not json'
    series_dict, output = main()
    with open(args.output_file, "w") as outfile:
        json.dump(output, outfile, indent=4)
    sys.stdout = default_stdout

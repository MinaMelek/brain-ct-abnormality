import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocessing import slice_preprocess, download_data
from tumor_predict import predict as tumor_predict
from stroke_predict import predict as stroke_predict, class_num as stroke_classes
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
    def __init__(self, target_files, dir_path, size, transform=None):
        self.target_files = target_files
        self.dir_path = dir_path
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        img_path = self.dir_path / self.target_files[idx]
        # Prepare image (slice) for prediction
        image = slice_preprocess(img_path, self.size)
        if self.transform:
            image = self.transform(image)
        return image


def main():
    patient_dir = download_data(args.url)
    i = 0
    series = {}
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
        # Only middle slices should be selected
        n_files = len(filenames)
        ignored_portion = n_files // 5
        target_files = filenames[ignored_portion: -ignored_portion]
        # TODO: redesign for efficiency
        new_size = (args.image_size, args.image_size)

        # PREDICTION
        batch_size = 64
        test_data = ImageDataset(target_files, dir_path, new_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=26)
        predict_1 = np.zeros(shape=(len(test_data), stroke_classes))
        predict_2 = np.zeros(shape=(len(test_data),))
        for j, images in enumerate(test_dataloader):
            p_1 = stroke_predict(images, model_path=os.path.join(args.model_dir, 'CTish_frac_model.pt'))
            p_2 = tumor_predict(images, model_path=os.path.join(args.model_dir, 'JUH_noisy_model.pt'))
            idx = j*batch_size
            predict_1[idx: idx+len(images)] = p_1
            predict_2[idx: idx+len(images)] = p_2
        series[i] = [dir_path, predict_1, predict_2]  # TODO: format the output
        i += 1
    return series


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gpu', dest='gpu', default=False, action='store_true',
                        help='load and predict on gpu, default: False (cpu)')
    parser.add_argument('-index', dest='gpu_index', type=str, default='0',
                        help='gpu index if you have multiple gpus, default: 0')
    parser.add_argument('-parallel', dest='parallel', default=False, action='store_true',
                        help='use model in parallel mode in case of multiple devices, default: False')
    parser.add_argument('-m-dir', dest='model_dir', type=str, default=os.path.join('..', 'models'),
                        help='select pytorch models directory, default: "../models/"')
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

    default_stdout = sys.stdout
    sys.stdout = Logger()
    series_dict = main()
    sys.stdout = default_stdout

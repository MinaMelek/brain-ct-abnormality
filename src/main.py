import os
import sys
from pathlib import Path
import numpy as np
from preprocessing import slice_preprocess, download_data
from tumor_predict import tumor_predict
from stroke_predict import stroke_predict
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
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


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
        images = []
        for j, f in enumerate(target_files):
            file_name = dir_path / f
            print(f"\t{file_name}: ", end='\t')
            new_size = (args.image_size, args.image_size)
            # Prepare image (slice) for prediction
            image = slice_preprocess(file_name, new_size)
            images.append(image)
        images = np.array(images)
        # PREDICTION
        predict_1 = stroke_predict(images, model_path=os.path.join(args.model_dir, 'CTish_frac_model.pt'))
        predict_2 = tumor_predict(images, model_path=os.path.join(args.model_dir, 'JUH_noisy_model.pt'))
        series[i] = [dir_path, predict_1, predict_2]
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
    series_dict = main(args)
    sys.stdout = default_stdout

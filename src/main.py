import os
import sys
from pathlib import Path
# import numpy as np
import cupy as cp
from torch.utils.data import Dataset, DataLoader
from torch.utils.dlpack import from_dlpack
# import torch.multiprocessing as mp
from preprocessing import slice_preprocess, download_data
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
        return from_dlpack(image.toDlpack())

# +
# # Nvidia Dali
# from nvidia.dali.pipeline import pipeline_def
# import nvidia.dali.types as types
# import nvidia.dali.fn as fn
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

# data_dir = '../assets/JPG/'


# @pipeline_def(batch_size=64, num_threads=4, device_id=0)
# def get_dali_pipeline(data_dir, crop_size):
#     images, labels = fn.readers.file(file_root=data_dir, name="Reader")
#     # decode data on the GPU
#     images = fn.decoders.image_random_crop(images, device="mixed", output_type=types.RGB)
#     # the rest of processing happens on the GPU as well
#     images = fn.resize(images, resize_x=crop_size, resize_y=crop_size)
#     images = fn.crop_mirror_normalize(images,
#                                       mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                       std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
#                                       mirror=fn.random.coin_flip())
#     return images, labels


# train_data = DALIGenericIterator(
#     [get_dali_pipeline(data_dir, 256)],
#     ['data', 'label'],
#     reader_name='Reader'
# )
# """
# [Warning]: File i0000010_0_0.dcm has extension that is not supported by the decoder. 
# Supported extensions: .flac, .ogg, .wav, .jpg, .jpeg, .png, .bmp, .tif, .tiff, .pnm, .ppm, .pgm, .pbm, .jp2, .webp, 
# """
# for x in train_data:
#     print(x[0]['label'].shape, x[0]['data'].shape)

# +
# fn.python_function
# -

def main():
    patient_dir = download_data(args.url)
    i = 0
    series = {}
    output = {}
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
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  #, pin_memory=True, num_workers=24)
        # preload to remove overhead
        tumor_model = load_model(os.path.join(args.model_dir, 'JUH_noisy_model.pt'),
                                 gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)
        predict_1 = cp.zeros(shape=(len(test_data), len(stroke_classes)))
        predict_2 = cp.zeros(shape=(len(test_data),))
        for j, images in enumerate(test_dataloader):
            p_1 = stroke_predict(images, model_path=os.path.join(args.model_dir, 'CTish_frac_model.pt'),
                                 gpu=args.gpu, parallel=args.parallel, gpu_index=args.gpu_index)
            p_2 = tumor_predict(images, tumor_model, gpu=args.gpu)  # TODO: remake for stroke, use args for inputs
            idx = j*batch_size
            predict_1[idx: idx+len(images)] = cp.asarray(p_1)
            predict_2[idx: idx+len(images)] = cp.asarray(p_2)
        series[i] = [dir_path, predict_1, predict_2]
        # format stroke prediction
        out_1 = dict(zip(stroke_classes, map(lambda x: f"{x * 100:.2f}%", predict_1.mean(0))))  # Todo: Change mean
        # format tumor prediction
        tumor_mean = (predict_2 > 0.5).mean()  # percentage of tumor slices
        from collections import defaultdict
        oc, h = defaultdict(lambda x=0: x), 0
        for itm in predict_2:
            if itm > 0.5:
                oc[h] += 1
            else:
                h += 1
        max_occ = max(oc.values())  # max number of consecutive slices predicted as tumor
        out_2 = {'Tumor': f"{predict_2[predict_2 > 0.5].mean() * 100:.2f}%" if max_occ > 3 and tumor_mean > 0.3 else
                          f"{predict_2[predict_2 <= 0.5].mean() * 100:.2f}%"}
        output[f"series_{i}"] = {'Hemorrhage': out_1, "Tumor": out_2}  # Todo:Rethink the structure

        i += 1
    return series, output


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

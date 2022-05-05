# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:49:06 2022

@author: Mina
"""

# In['Libraries']

import os
import sys
from pathlib import Path
import tempfile
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from prepare_dicom import prep_pipeline
from preprocessing import remove_noise, zoom_on_image, window
from tumor_predict import tumor_predict, standardize
# from tqdm import tqdm


# In['Preprocessing]
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


def download_data(url):  # TODO: download actual data
    """
    Placeholder
    
    1.	Receive the Dicom data URL.
    2.	Download Dicom data files.
    """
    
    patient_dir = f'../../CT brain/p_{url}'
    return patient_dir


def data_prep(url, saved=False, image_path=None):
    """
    3.	Prepare Dicom slices for predictions:
    """
    
    new_size = (256, 256)
    
    patient_dir = download_data(url)
    print('-'*115)
    print(f"patient-{url}")
    i = 0
    series_dict = {}
    for dirpath, dirnames, filenames in os.walk(patient_dir):
        # Filter to Dicom image only.
        filenames = [fi for fi in filenames if fi.endswith(".dcm")]
        # Ignore paths with no dicom files.
        if len(filenames) == 0 or len(dirnames) > 0:
            continue
        
        """
        a.	Determine patient-id/name, study-id and series-id.
        """
        dirpath = Path(dirpath)
        # TODO: Decide how to use these ids
        study_id, series_id = dirpath.parts[-2:]
        print(f"{dirpath}: Study: {study_id}, Series: {series_id}")
        
        """
        b.	Make a decision on which slices should be selected.
        """
        n_files = len(filenames)
        ignored_portion = n_files//5
        images = []
        for j, f in enumerate(filenames[ignored_portion: -ignored_portion]):
            file_name = dirpath / f
            print(f"\t{file_name}: ", end='\t')
            # Read Dicom file.
            img_dcm = pydicom.dcmread(file_name)

            """
            c.	Make sure the selected study is for a brain.
            """
            try:  # TODO: remove after testing
                if img_dcm.BodyPartExamined.lower() != 'head':
                    # print(img_dcm.BodyPartExamined, dirpath)
                    print('not a head!!')
                    break
            except AttributeError:
                if img_dcm.StudyDescription != 'BRAIN':
                    try:
                        if img_dcm.FilterType.lower() == 'body filter' and \
                                img_dcm.SeriesDescription != 'BRAIN WITHOUT CONTRAST':
                            # print(img_dcm.FilterType, dirpath)
                            print('not a head!!')
                            break
                    except AttributeError:
                        # print('######### no filter found', dirpath)
                        print('not a head!!')
                        break
            # print()
            # continue  # TODO: remove this

            """
            d.	Apply window and slope information from Dicom metadata.
            """
            try:
                image = prep_pipeline(img=img_dcm)
            except AttributeError as e:
                # If Dicom windowing information were unavailable.
                if 'Window' in str(e).split()[-1]:
                    print("didn't find window attributes")
                    image = window(img_dcm.pixel_array, w_level=40, w_width=120)
                else:
                    raise e

            """
            e.	Remove noise from the CT slice 
            """
            image = remove_noise(image)

            """
            f.	Center and zoom into the brain.
            """
            image = zoom_on_image(image, new_size)

            """
            g.	Save data.
            """
            if saved:
                if image_path is None:
                    image_path = Path(tempfile.mkdtemp())
                    print(f"saving inside a temp directory: {image_path}")
                im_path = str(image_path / f'{Path(patient_dir).name}_{i}_{j:03}.png')
                plt.imsave(im_path, image, cmap='gray')

            """
            PREDICTION
            """
            # Convert to rgb image
            image = np.concatenate([image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]], 2)
            # standardize images
            image = standardize(image)
            images.append(image)
        images = np.array(images)
        preds = tumor_predict(images)
        series_dict[i] = [dirpath, preds]
        i += 1
    return series_dict


# In[]
if __name__ == '__main__':
    default_stdout = sys.stdout
    sys.stdout = Logger()
    series_dict = data_prep(333)
    sys.stdout = default_stdout

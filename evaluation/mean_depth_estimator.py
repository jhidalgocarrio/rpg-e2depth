import numpy as np
import torch
import argparse
import glob
from os.path import join
import tqdm
import cv2

from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def FLAGS():
    parser = argparse.ArgumentParser("""Avg/Median Depth Estimator.""")

    # training / validation dataset
    parser.add_argument("--input_folder", type=str, default="", required=True)
    parser.add_argument("--output_folder", type=str, default="", required=True)

    flags = parser.parse_args()

    return flags

if __name__ == "__main__":
    flags = FLAGS()

    # predicted labels
    input_files = sorted(glob.glob(join(flags.input_folder, 'data', '*.npy')))

    num_it = len(input_files)
    all_tensor = None
    for idx in tqdm.tqdm(range(num_it)):
        input_depth = torch.from_numpy(np.load(input_files[idx])).to("cuda")
        input_depth = input_depth.expand((1, input_depth.shape[0], input_depth.shape[1]))
        if all_tensor is None:
            all_tensor = input_depth
        else:
            all_tensor =  torch.cat((all_tensor, input_depth), axis=0)


    mean_depth_prediction = all_tensor.mean(dim=0)
    median_depth_prediction = torch.median(all_tensor, dim=0).values

    np.save(join(flags.output_folder, "mean_depth_prediction.npy"), mean_depth_prediction.cpu().numpy())
    np.save(join(flags.output_folder, "median_depth_prediction.npy"), median_depth_prediction.cpu().numpy())
    print(all_tensor.shape)
import numpy as np
import torch
import argparse
import glob
from os.path import join
import tqdm
import cv2

from pprint import pprint



def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--target_dataset", default="", required=True)
    parser.add_argument("--predictions_dataset", default="", required=True)
    parser.add_argument("--event_masks", default="", required=True)
    parser.add_argument("--crop_target_ymax", default=200, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--prediction_offset", type=int, default=0)
    parser.add_argument("--target_offset", type=int, default=0)

    flags = parser.parse_args()

    return flags

metrics_keywords = [
    f"_threshold_delta_1.25",
    f"_threshold_delta_1.25^2",
    f"_threshold_delta_1.25^3",
    f"_abs_rel_diff",
    f"_squ_rel_diff",
    f"_RMS_linear",
    f"_RMS_log",
    f"_SILog",
    f"_mean_depth_error",
    f"_10_threshold_delta_1.25",
    f"_10_threshold_delta_1.25^2",
    f"_10_threshold_delta_1.25^3",
    f"_10_abs_rel_diff",
    f"_10_squ_rel_diff",
    f"_10_RMS_linear",
    f"_10_RMS_log",
    f"_10_SILog",
    f"_10_mean_depth_error",
    f"_20_threshold_delta_1.25",
    f"_20_threshold_delta_1.25^2",
    f"_20_threshold_delta_1.25^3",
    f"_20_abs_rel_diff",
    f"_20_squ_rel_diff",
    f"_20_RMS_linear",
    f"_20_RMS_log",
    f"_20_SILog",
    f"_20_mean_depth_error",
    f"_30_threshold_delta_1.25",
    f"_30_threshold_delta_1.25^2",
    f"_30_threshold_delta_1.25^3",
    f"_30_abs_rel_diff",
    f"_30_squ_rel_diff",
    f"_30_RMS_linear",
    f"_30_RMS_log",
    f"_30_SILog",
    f"_30_mean_depth_error",
    f"event_masked_threshold_delta_1.25",
    f"event_masked_threshold_delta_1.25^2",
    f"event_masked_threshold_delta_1.25^3",
    f"event_masked_abs_rel_diff",
    f"event_masked_squ_rel_diff",
    f"event_masked_RMS_linear",
    f"event_masked_RMS_log",
    f"event_masked_SILog",
    f"event_masked_mean_depth_error",
    f"event_masked_10_threshold_delta_1.25",
    f"event_masked_10_threshold_delta_1.25^2",
    f"event_masked_10_threshold_delta_1.25^3",
    f"event_masked_10_abs_rel_diff",
    f"event_masked_10_squ_rel_diff",
    f"event_masked_10_RMS_linear",
    f"event_masked_10_RMS_log",
    f"event_masked_10_SILog",
    f"event_masked_10_mean_depth_error",
    f"event_masked_30_threshold_delta_1.25",
    f"event_masked_30_threshold_delta_1.25^2",
    f"event_masked_30_threshold_delta_1.25^3",
    f"event_masked_30_abs_rel_diff",
    f"event_masked_30_squ_rel_diff",
    f"event_masked_30_RMS_linear",
    f"event_masked_30_RMS_log",
    f"event_masked_30_SILog",
    f"event_masked_30_mean_depth_error",
    f"event_masked_20_threshold_delta_1.25",
    f"event_masked_20_threshold_delta_1.25^2",
    f"event_masked_20_threshold_delta_1.25^3",
    f"event_masked_20_abs_rel_diff",
    f"event_masked_20_squ_rel_diff",
    f"event_masked_20_RMS_linear",
    f"event_masked_20_RMS_log",
    f"event_masked_20_SILog",
    f"event_masked_20_mean_depth_error",
]

def add_to_metrics(metrics, target_, prediction_, mask, prefix="", debug = False):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    prediction_ = np.clip(prediction_, 0, 102)
    depth_mask = (target_ > 0) & (target_ < 101)
    mask = mask & depth_mask
    eps = 1e-5

    target = target_[mask]
    prediction = prediction_[mask]

    # thresholds
    ratio = np.max(np.stack([target/(prediction+eps),prediction/(target+eps)]), axis=0)

    new_metrics = {}

    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)
    
    # abs diff
    log_diff = np.log(target+eps)-np.log(prediction+eps)
    abs_diff = np.abs(target-prediction)
    
    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff/(target+eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2/(target**2+eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff**2).mean()-(log_diff.mean())**2
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()

    for k, v in new_metrics.items():
        metrics[k] += v

    if debug:
        if debug:
            pprint(new_metrics)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(ncols=3, nrows=4)
            ax[0, 0].imshow(np.log(target_+eps),vmin=0,vmax=np.log(1000))
            ax[0, 0].set_title("log target")
            ax[0, 1].imshow(np.log(prediction_+eps),vmin=0,vmax=np.log(1000))
            ax[0, 1].set_title("log prediction")
            ax[0, 2].imshow(np.max(np.stack([target_ / (prediction_ + eps), prediction_ / (target_ + eps)]), axis=0))
            ax[0, 2].set_title("max ratio")
            ax[1, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)))
            ax[1, 0].set_title("abs log diff")
            ax[1, 1].imshow(np.abs(target_ - prediction_))
            ax[1, 1].set_title("abs diff")
            ax[1, 2].imshow(target_, vmin=0, vmax=200)
            ax[1, 2].set_title("target depth")

            log_diff_ = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
            log_diff_[~mask] = 0
            ax[2, 0].imshow(log_diff_)
            ax[2, 0].set_title("abs log diff masked")
            abs_diff_ = np.abs(target_ - prediction_)
            abs_diff_[~mask] = 0
            ax[2, 1].imshow(abs_diff_)
            ax[2, 1].set_title("abs diff masked")
            target_debug = target_.copy()
            target_debug[~mask] = 0
            ax[2, 2].imshow(target_debug, vmin=0, vmax=200)
            ax[2, 2].set_title("target depth masked")

            ax[3,0].hist(abs_diff_.reshape((-1,)), bins=np.arange(1000))
            ax[3,0].set_title("Abs error histogram")

            mx = np.max(abs_diff_)
            #print(np.where(abs_diff_> 0.9*mx))
            plt.show()

    return metrics


if __name__ == "__main__":
    flags = FLAGS()
    
    # predicted labels 
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, 'data', '*.npy')))
    prediction_files = prediction_files[flags.prediction_offset:]
    prediction_files = prediction_files[102+200:]
    target_files = sorted(glob.glob(join(flags.target_dataset, 'data', '*npy')))
    target_files = target_files[flags.target_offset:]
    if flags.event_masks is not "":
        event_frame_files = sorted(glob.glob(join(flags.event_masks, '*png')))
        event_frame_files = event_frame_files[flags.prediction_offset:]
    prediction_timestamps = np.genfromtxt(join(flags.predictions_dataset, 'data/timestamps.txt'))
    target_timestamps = np.genfromtxt(join(flags.target_dataset, 'data/timestamps.txt'))

    assert len(prediction_files)>0
    assert len(target_files)>0

    use_event_masks = len(event_frame_files)>0

    metrics = {}

    num_it = len(prediction_files)
    for idx in tqdm.tqdm(range(num_it)):
        p_file, t_file = prediction_files[idx], target_files[idx]
        predicted_depth = np.load(p_file)

        target_depth = np.load(t_file)
        target_depth = target_depth[:flags.crop_target_ymax]
        assert predicted_depth.shape == target_depth.shape

        event_mask = (np.ones_like(target_depth)>0)
        debug = flags.debug and idx == flags.idx
        metrics = add_to_metrics(metrics, target_depth, predicted_depth, event_mask, "_", debug=debug)

        for depth_threshold in [10, 20, 30]:
            depth_threshold_mask = (target_depth < depth_threshold)
            add_to_metrics(metrics, target_depth, predicted_depth, event_mask & depth_threshold_mask,
                           prefix=f"_{depth_threshold}_", debug=debug)

        if use_event_masks:
            ev_frame_file = event_frame_files[idx]
            event_frame = cv2.imread(ev_frame_file)
            event_mask = (np.sum(event_frame.astype("float32"), axis=-1)>0)
            assert event_mask.shape == target_depth.shape
            add_to_metrics(metrics, target_depth, predicted_depth, event_mask, prefix="event_masked_")

            for depth_threshold in [10, 20, 30]:
                depth_threshold_mask = target_depth < depth_threshold
                #$debug=True
                add_to_metrics(metrics, target_depth, predicted_depth, event_mask & depth_threshold_mask, prefix=f"event_masked_{depth_threshold}_", debug=debug)


        #if idx % 400 == 0:
        #    print("Intermediate metric:")
        #    pprint(metrics)

    pprint({k: v/num_it for k,v in metrics.items()})



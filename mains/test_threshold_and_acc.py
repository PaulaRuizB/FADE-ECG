import deepdish as dd
import numpy as np
import copy
from operator import itemgetter
import torch
from torchmetrics.functional.regression import mean_absolute_error
import sys
sys.path.append('../')
from models.SlowFast_raw_signal_nsr_big import slowfast_raw
from collections import OrderedDict
from functools import partial
from sklearn.model_selection import train_test_split
import math

def test_threshold(path_test_set, x_names, amp_autocast, y_names, dist_percentil, model, input_size_seconds):

    metrics_eval = [
        "mae_mae_norm_product",
    ]
    for metric in metrics_eval:
        globals()[metric + "_arr"] = []
        globals()[metric + "_nsr"] = []

    for idx, i in enumerate(x_names):

        image = dd.io.load(path_test_set + i + ".h5")


        input = torch.Tensor(image[0, :])
        label = torch.Tensor(image[1, :])

        with torch.no_grad():
            with amp_autocast():
                input = input[int((4 - input_size_seconds) * 128):]
                label = label[int((4 - input_size_seconds) * 128):]

                input = torch.unsqueeze(torch.unsqueeze(input, dim=0), dim=0).to(0).to(torch.float16)
                label = torch.unsqueeze(torch.unsqueeze(label, dim=0), dim=0).to(0).to(torch.float16)

                model = model.to(0).to(torch.float16)
                output = model(input)

                if type(output) == tuple:
                    output = output[1]

                mae_mae_norm_product = (mean_absolute_error(output[:, :, -128:],
                                                            label[:, :, -128:])) * compute_mean_norm(
                    output[:, :, -128:], label[:, :, -128:])

                if y_names[idx] == '(N':
                    globals()["mae_mae_norm_product_nsr"].append(mae_mae_norm_product.cpu())

                elif y_names[idx] != '(N':
                    globals()["mae_mae_norm_product_arr"].append(mae_mae_norm_product.cpu())

    for metric in metrics_eval:
        acc_max = 0
        acc_max_nsr = 0
        acc_max_arr = 0
        p_mse_max = 0.

        metric_arr = np.array(globals()[metric + "_arr"])
        metric_nsr = np.array(globals()[metric + "_nsr"])

        print("#################### {} METRICS ####################".format(metric))
        print("Num samples NSR {}".format(metric_nsr.shape[0]))
        print("Num samples ARR {}".format(metric_arr.shape[0]))
        for i in np.arange(0, 100.0, dist_percentil):

            p_metric_arr = np.percentile(metric_arr, i)

            acc_metric_nsr = (metric_nsr < p_metric_arr).sum() * 100.0 / metric_nsr.shape[0]

            acc_metric_arr = (metric_arr >= p_metric_arr).sum() * 100.0 / metric_arr.shape[0]

            acc_total = (acc_metric_nsr + acc_metric_arr) / 2.0

            if acc_total >= acc_max:
                acc_max = acc_total
                acc_max_nsr = acc_metric_nsr
                acc_max_arr = acc_metric_arr
                p_mse_max = p_metric_arr

        print("ACC: {:.2f}, ACC arr {:.2f}, ACC NSR {:.2f}, percentile {:.4f}".format(acc_max, acc_max_arr,
                                                                                            acc_max_nsr, p_mse_max))

    return acc_max, acc_max_arr, acc_max_nsr, p_mse_max

def test_acc(path_test_set, x_names, amp_autocast, y_names, dist_percentil, anomalies_mae_mae_norm_product, model, input_size_seconds):

    metrics_eval = [
        "mae_mae_norm_product",
    ]
    for metric in metrics_eval:
        globals()[metric + "_arr"] = []
        globals()[metric + "_nsr"] = []

    output_nsr = []
    output_an = []
    labels_an = []
    list_test_nsr = []
    list_test_an = []

    for idx, i in enumerate(x_names):

        image = dd.io.load(path_test_set + i + ".h5")


        input = torch.Tensor(image[0, :])
        label = torch.Tensor(image[1, :])

        with torch.no_grad():
            with amp_autocast():
                input = input[int((4 - input_size_seconds) * 128):]
                label = label[int((4 - input_size_seconds) * 128):]

                input = torch.unsqueeze(torch.unsqueeze(input, dim=0), dim=0).to(0).to(torch.float16)
                label = torch.unsqueeze(torch.unsqueeze(label, dim=0), dim=0).to(0).to(torch.float16)

                model = model.to(0).to(torch.float16)
                output = model(input)

                if type(output) == tuple:
                    output = output[1]

                # mae_mae_norm_product
                mae_mae_norm_product = (mean_absolute_error(output[:, :, -128:],
                                                            label[:, :, -128:])) * compute_mean_norm(
                    output[:, :, -128:], label[:, :, -128:])

                if y_names[idx] == '(N':
                    globals()["mae_mae_norm_product_nsr"].append(mae_mae_norm_product.cpu())
                    output_nsr.append(output)
                    list_test_nsr.append(i)

                elif y_names[idx] != '(N':
                    anomalies_mae_mae_norm_product[y_names[idx]].append(mae_mae_norm_product.cpu())
                    globals()["mae_mae_norm_product_arr"].append(mae_mae_norm_product.cpu())
                    output_an.append(output)
                    labels_an.append(y_names[idx])
                    list_test_an.append(i)


    for metric in metrics_eval:

        metric_arr = np.array(globals()[metric + "_arr"])
        metric_nsr = np.array(globals()[metric + "_nsr"])

        print("#################### {} METRICS ####################".format(metric))
        print("Num samples NSR {}".format(metric_nsr.shape[0]))
        print("Num samples ARR {}".format(metric_arr.shape[0]))

        acc_metric_nsr = (metric_nsr < dist_percentil).sum() * 100.0 / metric_nsr.shape[0]
        acc_metric_arr = (metric_arr >= dist_percentil).sum() * 100.0 / metric_arr.shape[0]

        acc_total = (acc_metric_nsr + acc_metric_arr) / 2.0

        print("ACC : {:.2f}, ACC arr {:.2f}, ACC NSR {:.2f}".format(acc_total, acc_metric_arr,
                                                                                            acc_metric_nsr))

    for key in anomalies_mae_mae_norm_product.keys():
        print("ACC mae_mae_norm_product {}: {:.2f}%".format(key, 100. * (
                np.sum(anomalies_mae_mae_norm_product[key] > dist_percentil) / len(
            anomalies_mae_mae_norm_product[key]))))

    return acc_total, acc_metric_arr, acc_metric_nsr, metric_arr, metric_nsr, output_nsr, output_an, labels_an, list_test_nsr, list_test_an

def compute_mean_norm(y_gt, y_pred):
    min_val = torch.min(y_gt)
    max_val = torch.max(y_gt)
    y_gt = (y_gt - min_val) /  (max_val - min_val)

    min_val = torch.min(y_pred)
    max_val = torch.max(y_pred)
    y_pred = (y_pred - min_val) /  (max_val - min_val)

    mean = torch.mean(torch.abs(y_gt-y_pred))
    return mean
def main(dropout, num_channels, add_FC, path_trained_model, path_datasetinfo, path_test_set,
         path_save_results, step_percentile, test_percentage, seed, input_size_seconds):

    print(path_trained_model)
    print("loading------------------")
    print(seed)

    ##DATA
    dataset_info = dd.io.load(path_datasetinfo)
    labels = dataset_info["labels"]
    list_names = dataset_info["list_names"]
    train_test_label = dataset_info["train_test_label"]

    anomalies_, number = np.unique(labels, return_counts=True)
    anomalies = {}
    for key in anomalies_:
        anomalies[key] = []

    anomalies_mae_mae_norm_product = copy.deepcopy(anomalies)

    positions_arr_nsr = [idx for idx, (label, sample) in
                             enumerate(zip(train_test_label, list_names)) if
                             label == 0]


    list_names_arr_nsr = list(itemgetter(*list(positions_arr_nsr))(list_names))
    labels_arr_nsr = list(itemgetter(*list(positions_arr_nsr))(labels))

    ##MODEL
    args.device = 0
    device = torch.device(args.device)
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    model = slowfast_raw(dropout=dropout, num_channels=num_channels, layer_norm=False,
                             stochastic_depth=0, add_FC_BN=False, remove_BN=False, add_FC=add_FC,
                             replaceBN_LN=False, add_FC_LN=False, add_2FC=False, input_size_seconds=input_size_seconds,
                             slowfast_output=False, unet_output=True)

    state_dict = torch.load(path_trained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:9]=="_orig_mod":
            name = k[10:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        else:
            new_state_dict = state_dict
            break

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    #######

    dist_percentil = step_percentile
    min_samples = math.ceil(1/test_percentage)

    invalid_classes = anomalies_[number<min_samples]
    mask_final_test = np.isin(labels_arr_nsr, invalid_classes)
    X_final_test = np.array(list_names_arr_nsr)[mask_final_test]
    Y_final_test = np.array(labels_arr_nsr)[mask_final_test]

    valid_classes = anomalies_[number >= min_samples]
    mask = np.isin(labels_arr_nsr, valid_classes)
    X_filtered = np.array(list_names_arr_nsr)[mask]
    y_filtered = np.array(labels_arr_nsr)[mask]

    x_test_idx, x_th_idx, y_test_labels, y_th_labels = train_test_split(X_filtered, y_filtered, test_size=test_percentage, stratify=y_filtered, random_state=seed)

    x_test = np.concatenate((X_final_test, x_test_idx))
    y_test = np.concatenate((Y_final_test, y_test_labels))

    ### TEST TH
    acc_max, acc_max_arr, acc_max_nsr, p_mse_max = test_threshold(path_test_set, x_th_idx, amp_autocast, y_th_labels, dist_percentil, model, input_size_seconds)

    ### TEST ACC
    acc_total, acc_metric_arr, acc_metric_nsr, metric_arr, metric_nsr, output_nsr, output_an, labels_an, list_test_nsr, list_test_an = test_acc(path_test_set, x_test, amp_autocast, y_test, p_mse_max, anomalies_mae_mae_norm_product, model, input_size_seconds)
    an, count = np.unique(y_test, return_counts=True)
    test_samples = dict(zip(an, count))

    print(test_samples)
    accuracy = {'list_images_th': x_th_idx,
                'labels_th': y_th_labels,
                'list_images_test': x_test,
                'labels_test': y_test,
                'acc_th': acc_max,
                'acc_max_arr_th': acc_max_arr,
                'acc_max_nsr_th': acc_max_nsr,
                'th': p_mse_max,
                'acc_final_test': acc_total,
                'acc_final_arr': acc_metric_arr,
                'acc_final_nsr': acc_metric_nsr,
                'percentage': test_percentage,
                'step_percentile': step_percentile,
                'dist_arr': metric_arr,
                'dist_nsr': metric_nsr,
                'output_nsr': output_nsr,
                'output_an': output_an,
                'labels_an': labels_an,
                'list_test_nsr': list_test_nsr,
                'list_test_an': list_test_an
    }

    dd.io.save(path_save_results + "results.h5".format(acc_total, dist_percentil, seed), accuracy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')

    parser.add_argument('--dropout', type=float, required=False, default=0, help='Dropout rate applied to fusion layers.')
    parser.add_argument('--num_channels', type=int, required=False, default=1, help='Number of input channels')
    parser.add_argument('--add_FC', default=True, action='store_true', help="Add an additional fully connected (FC) layer at the end of the model")
    parser.add_argument('--path_trained_model', default='', type=str, required=False, help="Path to the pretrained model to load.")
    parser.add_argument('--path_datasetinfo', default='', type=str, required=False, help="Path to the dataset metadata file (HDF5 with signal information).")
    parser.add_argument('--path_test_set', default='', type=str, required=False, help='Path to the directory containing the test dataset.')
    parser.add_argument('--path_save_results', default='', type=str, required=False, help="Directory where results will be saved.")
    parser.add_argument('--step_percentile', type=float, required=False, default=0.0001, help='Step size (as percentile) used in sliding window evaluation.')
    parser.add_argument('--test_percentage', type=float, required=False, default=0.2, help='Fraction of the dataset to use for testing.')
    parser.add_argument('--seed', type=int, required=False, default=62, help='Random seed used for test reproducibility.')
    parser.add_argument('--input_size_seconds', default=4, type=float, help='Duration (in seconds) of each input sample.')

    args = parser.parse_args()
    dropout = args.dropout
    num_channels = args.num_channels
    add_FC = args.add_FC
    path_trained_model = args.path_trained_model
    path_datasetinfo = args.path_datasetinfo
    path_test_set = args.path_test_set
    path_save_results = args.path_save_results
    step_percentile = args.step_percentile
    test_percentage = args.test_percentage
    seed = args.seed
    input_size_seconds = args.input_size_seconds

    main(dropout, num_channels, add_FC, path_trained_model, path_datasetinfo, path_test_set, path_save_results, step_percentile, test_percentage, seed, input_size_seconds)

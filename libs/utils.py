import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from sklearn import metrics
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from collections import defaultdict
import argparse
import math


def get_single_mask(surface, ring_diameter, center_slice):
    y_spacing = 0.01141 # 0.010837
    x_spacing = 0.120782 # 0.11544
    ring_radius = ring_diameter / 2
    mask = surface.copy()
    h, w = surface.shape
    center_y = w // 2
    rr, cc = ellipse(center_slice, center_y, ring_radius / x_spacing, ring_radius / y_spacing)
    rr = np.clip(rr, -h + 1, h - 1)
    cc = np.clip(cc, -w + 1, w - 1)
    mask[rr, cc] = 1
    return mask


def generate_mask(ring_diameters, slices, h, w, center_slice, surface=None, channels='channels_last'):
    if surface is None:
        surface = np.zeros((slices, w))

    masks = []
    for diam in ring_diameters:
        masks.append(get_single_mask(surface, diam, center_slice))

    rings = []

    for i in range(len(masks)):
        if i == 0:
            mask_ring = masks[i]
        else:
            mask_ring = masks[i] * (1 - masks[i - 1])

        mask_ring = mask_ring.astype(np.uint8)
        if channels == 'channels_last':
            mask_ring = np.moveaxis(mask_ring, 0, -1)
            rings.append(np.repeat(mask_ring[np.newaxis, :, :], h, axis=0))
        else:
            rings.append(np.repeat(mask_ring[:, np.newaxis, :], h, axis=1))

    return rings

def resize_rings(rings, new_size, n_channels, channels='channels_last'):
    if isinstance(new_size, int):
        new_size = (new_size, new_size, n_channels) if channels=='channels_last' else (n_channels, new_size, new_size)

    for ring in rings:
        ring.resize(new_size, refcheck=False)

    return rings

def calculate_metrics(y_true, y_pred):
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    y_pred_sigmoid = sigmoid(y_pred)
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred_sigmoid, multi_class='ovr', average='samples')
    except ValueError as e:
        print('Error with ROC AUC', str(e))
        roc_auc = 0

    mAP = metrics.average_precision_score(y_true, y_pred_sigmoid, average='micro')
    MAP = metrics.average_precision_score(y_true, y_pred_sigmoid, average='macro')

    prediction_int = np.zeros_like(y_pred_sigmoid)
    prediction_int[y_pred_sigmoid > 0.5] = 1

    mf1 = metrics.f1_score(y_true, prediction_int, average='micro')
    Mf1 = metrics.f1_score(y_true, prediction_int, average='macro')

    dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1}
    return dict_metrics

def calculate_metrics_sigmoid(y_true, y_pred):
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='samples')
    except ValueError as e:
        print('Error with ROC AUC', str(e))
        roc_auc = 0

    mAP = metrics.average_precision_score(y_true, y_pred, average='micro')
    MAP = metrics.average_precision_score(y_true, y_pred, average='macro')

    prediction_int = np.zeros_like(y_pred)
    prediction_int[y_pred > 0.5] = 1

    mf1 = metrics.f1_score(y_true, prediction_int, average='micro', zero_division=1)
    Mf1 = metrics.f1_score(y_true, prediction_int, average='macro', zero_division=1)

    dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1}
    return dict_metrics

def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def normalize(x, **kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    try:
        channels = kwargs['channels']
    except KeyError:
        channels = 'first'
    if isinstance(mean, list):
        mean = np.array(mean)
        std = np.array(std)
        if channels == 'first':
            mean = mean[np.newaxis, :, np.newaxis, np.newaxis] # We apply the transformation in the cannels
            std = std[np.newaxis, :, np.newaxis, np.newaxis]  # We apply the transformation in the cannels (BATCH, CHANNEL, WIDTH, HEIGHT)
        else:
            mean = mean[np.newaxis, np.newaxis, np.newaxis, :]  # We apply the transformation in the cannels
            std = std[np.newaxis, np.newaxis, np.newaxis, :]  # We apply the transformation in the cannels (BATCH, CHANNEL, WIDTH, HEIGHT)
    x = np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    return (x - mean) / std

def normalize_01(x, **kwargs):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def append_to_dict(x, y, mode='append'):
    '''
    Append dict x to y. (Y is the biggest)
    '''
    for key in y.keys():
        if 'irf' in key or 'srf' in key:
            val = np.clip(x[key], 0, 1)
        else:
            val = x[key]

        if mode == 'append':
            y[key].append(val)
        elif mode == 'extend':
            y[key].extend(val)
        else:
            raise ValueError('Mode not supported')

    return y


def df_to_plot(df, column_x, column_y, dataset_name):
    df.plot(x=column_x, y=column_y, kind='scatter')
    name = column_y.replace(' ', '_')
    if dataset_name != '':
        plt.savefig(f'results/Figures/{dataset_name}/{name}.png')
    else:
        plt.savefig(f'results/Figures/{name}.png')

def restart_vectors(MAX_BATCH_SIZE, N_PIXEL_H, N_PIXEL_W):
    i=0
    oct_slices = np.zeros((MAX_BATCH_SIZE, N_PIXEL_H, N_PIXEL_W))
    filenames_batch = []
    bscans_batch = []

    return i, oct_slices, filenames_batch, bscans_batch

def swap_keys_dict(d):
    r = defaultdict(dict)
    for pkey in d.keys():
        for skey, items in d[pkey].items():
            r[skey][pkey] = items

    return r

def get_label_features(true, pred, threshold=None):
    if threshold is not None:
        pred = pred > threshold
    else:
        assert int(pred) == pred, 'A threshold should be provided for non-int values of pred.'

    if true == 1:
        if true == pred:
            return 'TP'
        else:
            return 'FN'
    elif true == 0:
        if true == pred:
            return 'TN'
        else:
            return 'FP'

def max_minibatch_size(model_name):
    batches_dict = {
        # Coefficients:
        'efficientnet-b0': 128,
        'efficientnet-b1': 128,
        'efficientnet-b2': 64,
        'efficientnet-b3': 64,
        'efficientnet-b4': 8,
        'efficientnet-b5': 8,
        'efficientnet-b6': 8,
        'efficientnet-b7': 4,
        'efficientnet-b8': 4,
        'efficientnet-l2': 4,
    }
    return batches_dict[model_name]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_one_hot(number, shape):
    assert len(shape) == 2, 'Shape must be BATCH_SIZE, CLASSES'
    r = torch.zeros(shape)
    r[:, number] = 1
    return r

def plot_roc(fpr, tpr, roc_auc, biomarker_name, title, filename, subtitle=None):
    if subtitle is None:
        subtitle = f'ROC curve (area = %0.2f) {biomarker_name}' % roc_auc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=subtitle)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename.replace('png', 'eps'))
    plt.close()

def plot_precision_recall(precision, recall, ap, biomarker_name, title, filename, subtitle=None):
    if subtitle is None:
        subtitle = f'AP = %0.2f {biomarker_name}' % ap
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label=subtitle)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.savefig(filename)
    plt.savefig(filename.replace('png', 'eps'))
    plt.close()

def plot_roc_group(fpr, tpr, roc_auc, names):
    lw = 2
    for label in fpr.keys():
        plt.figure()
        for i, mask_type in enumerate(fpr[label].keys()):
            plt.plot(fpr[label][mask_type], tpr[label][mask_type], lw=lw,
                     label=f'ROC {names[i]} (area = %0.2f)' % roc_auc[label][mask_type])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC {label}')
        plt.legend(loc="lower right")
        plt.savefig(f'results/Figures/ROC/roc_{label}.png')

def plot_ap_group(recall, precision, ap, names):
    lw = 2
    for label in recall.keys():
        plt.figure()
        for i, mask_type in enumerate(recall[label].keys()):
            plt.plot(recall[label][mask_type], precision[label][mask_type], lw=lw,
                     label=f'AP {names[i]}= %0.2f' % ap[label][mask_type])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision Recall {label}')
        plt.legend(loc="lower right")
        plt.savefig(f'results/Figures/AP/ap_{label}.png')

def txt_to_list(filename):
    with open(filename) as f:
        content = [line.rstrip('\n') for line in f]
    return content


def get_present_rings(bscan_number_batch, centers, onehot=False, return_tensor=False):
    if onehot:
        bscan_number_batch = torch.argmax(bscan_number_batch, dim=1)

    rings = torch.zeros((bscan_number_batch.shape[0], 3))  # 1mm, 3mm, 6mm
    rings[:, -1] = 1

    for i, sample in enumerate(bscan_number_batch):
        if centers[i] - 12 <= sample <= centers[i] + 12:
            rings[i, 1] = 1
        if centers[i] - 4 <= sample <= centers[i] + 4:
            rings[i, 0] = 1

    if return_tensor:
        return torch.as_tensor(rings)
    else:
        return rings.numpy()

def join_labels(x, x_srf, x_irf):
    """
    They have to be numpy arrays
    """
    temp = np.append(x, x_srf, axis=1)
    temp = np.append(temp, x_irf, axis=1)
    return temp

def get_lines_sorted(mask):
    lines = []
    for i_ring, ring in enumerate(mask):
        for i in range(1, len(mask[i_ring])):
            if mask[i_ring][i] != mask[i_ring][i - 1]:
                lines.append(i)
    lines.append(len(mask[0]))
    lines = sorted(set(lines))
    return lines


def convert_to_rings(p, slice_numbers, centers, n_pixel_w, n_pixel_h, ring_diameters, col_w):
    predicted_rings = np.empty((p.shape[0], 3))
    for i_slice, slice_number in enumerate(slice_numbers):
        if centers[i_slice] == -1:
            predicted_rings[i_slice] = [-100, -100, -100]
            continue
        mask_generator = generate_mask(ring_diameters, 49, n_pixel_h, n_pixel_w, centers[i_slice])
        mask = [mask_generator[i][0, :, slice_number] for i in range(len(mask_generator))]
        present_rings = get_present_rings(np.array([slice_number]), np.array([centers[i_slice]]), return_tensor=False)

        start_ring = 0
        res = []

        for i in get_lines_sorted(mask):
            end_ring = math.ceil(i / col_w)
            try:
                res.append(np.max(p[i_slice, start_ring-1:end_ring + 1])) # -1 because of the ceil before.
            except ValueError:
                res.append(p[i_slice, -1])
            start_ring = end_ring

        predicted_rings[i_slice] = combine_rings(res, present_rings)

    return predicted_rings

def combine_rings(p_per_region, present_rings):
    if len(p_per_region) > 1:
        if np.sum(present_rings) == 1:
            predicted_rings = [0, 0, p_per_region[1]]
        elif np.sum(present_rings) == 2:
            if len(p_per_region) == 5:
                predicted_rings = [0, p_per_region[2], max(p_per_region[1], p_per_region[3])]
            else:
                predicted_rings = [0, p_per_region[1], max(p_per_region[0], p_per_region[2])]
        else:
            predicted_rings = [p_per_region[2], max(p_per_region[1], p_per_region[3]), max(p_per_region[0], p_per_region[-1])]
    else:
        predicted_rings = [0,0,0]


    return predicted_rings

slices_present = {'mm1': [20, 28], 'mm3': [12, 36]}  # No mm6 because all slices have a part inside 6mm ring

def nonpresent_to_nan(series):
    ring = series.name.split('_')[0]
    if ring != 'present' and ring != 'mm6':
        bscans = series.index.get_level_values('bscan')
        series[(bscans <= slices_present[ring][0]) | (bscans >= slices_present[ring][1])] = np.nan
        # series[(series[bscans >= slices_present[ring][0]]) & (series[bscans <= slices_present[ring][1]])] = np.nan
    return series

def remove_outer_slices(df, name):
    ring = name.split('_')[0]

    if ring in slices_present.keys():
        df = df[(df['level_1'] >= slices_present[ring][0]) & (df['level_1'] <= slices_present[ring][1])]
    true_labels = df[name + '_true']
    pred_labels = df[name + '_pred']

    return true_labels, pred_labels

def remove_outer_slices_base(df, name):
    ring = name.split('_')[0]

    if ring in slices_present.keys():
        df = df[(df['level_1'] >= slices_present[ring][0]) & (df['level_1'] <= slices_present[ring][1])]
    true_labels = df[name + '_true']
    pred_labels = df[name + '_pred']
    base_labels = df[name + '_base']

    return true_labels, pred_labels, base_labels

def remove_outer_slices_diff(df, name):
    df = df[[name + '_true', name + '_pred']]
    df = df[df[name + '_pred'] != 0]
    df = df[df[name + '_pred'] != -100]

    true_labels = df[name + '_true']
    pred_labels = df[name + '_pred']
    return true_labels, pred_labels

def np_tv_loss(img, weight=1):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=(0, 1))
    bs_img, c_img, h_img, w_img = img.shape
    tv_h = np.power(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = np.power(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h.item() + tv_w.item()) / (bs_img * c_img * h_img * w_img)

def convert_to_rings_weighted(p_columns, slice_numbers, centers, n_pixel_w, n_pixel_h, ring_diameters, col_w):
    predicted_rings = np.empty((p_columns.shape[0], 3))
    for i_slice, slice_number in enumerate(slice_numbers):
        if centers[i_slice] == -1:
            predicted_rings[i_slice] = [-100, -100, -100]
            continue
        mask_generator = generate_mask(ring_diameters, 49, n_pixel_h, n_pixel_w, centers[i_slice])
        mask = [mask_generator[i][0, :, slice_number] for i in range(len(mask_generator))]
        present_rings = get_present_rings(np.array([slice_number]), np.array([centers[i_slice]]), return_tensor=False)

        lines = get_lines_sorted(mask)
        devolver = []
        acaba = [math.ceil(prima/col_w)-1 for prima in lines]
        empieza = [0] + acaba[:-1]
        cuanto = [(prima % col_w) / col_w for prima in lines]

        for i in range(len(lines)):
            default = [0] * 16
            cuanto[i] = cuanto[i] if cuanto[i] != 0 else 1
            if empieza[i] < acaba[i]:
                default[(empieza[i] + 1):(acaba[i])] = [1] * (acaba[i] - 1 - empieza[i])
                default[empieza[i]] = 1 - cuanto_hay(devolver, empieza[i])
                default[acaba[i]] = cuanto[i]
            elif acaba[i] == empieza[i]:
                default[acaba[i]] = cuanto[i] - cuanto_hay(devolver, acaba[i])
            devolver.append(default)

        devolver = np.array(devolver)
        p_rings = p_columns[i_slice] * devolver # This is (#regions, #columns)
        res = p_rings.max(axis=1)

        predicted_rings[i_slice] = combine_rings(res, present_rings)

    return predicted_rings


def cuanto_hay(devolver, i):
    try:
        return sum([lista[i] for lista in devolver])
    except:
        return 0

def fill_segmentation_uint(data, to_fill, names, slice_numbers):
    H, W = data.shape[1:]
    for i, (name, slice_number) in enumerate(zip(names, slice_numbers)):
        if name not in to_fill.keys():
            to_fill[name] = np.empty((49, H, W), dtype=np.uint8)
        to_fill[name][slice_number] = data[i]
    return to_fill

def fill_segmentation_pixcol(data, to_fill, names, slice_numbers):
    C, W = data.shape[1:]
    for i, (name, slice_number) in enumerate(zip(names, slice_numbers)):
        if name not in to_fill.keys():
            to_fill[name] = np.empty((49, C, W))
        to_fill[name][slice_number] = data[i]
    return to_fill
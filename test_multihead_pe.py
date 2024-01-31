import torch
from libs.data_retriever_encoding import OCTSliceMaskDataset, CLAHE, Resize
import torchvision.transforms as transforms
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
import argparse
from libs.models_pe import MultiHeadEfficientNet_PE
import numpy as np
import pandas as pd
import torch.nn as nn


def generate_labels():
    N_PIXEL_H = 496
    N_PIXEL_W = 512
    pix_col = N_PIXEL_W / num_columns
    ring_diameters = [1, 3, 6]

    model_name = f'efficientnet-b{model_number}'
    full_model_path = Path(__file__).resolve().parents[0].joinpath(model_path)
    model_date = model_path.split('/')[-1].split('.')[0].split('_')[1]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tifs = data_path.joinpath('extractedTifs')
    labels_path = data_path.joinpath('BSCAN')

    minibatch_size = u.max_minibatch_size(model_name)

    # mask_generator = u.generate_mask(ring_diameters, 49, N_PIXEL_H, N_PIXEL_W, 24)

    if use_clahe:
        t = transforms.Compose([CLAHE(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset = OCTSliceMaskDataset(mode='all', image_set=tifs,
                                  srf_gt_filename=labels_path.joinpath('srf_location.csv'),
                                  irf_gt_filename=labels_path.joinpath('irf_location.csv'), transform_image=t,
                                  encoding_type='noencoding')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=False, num_workers=1)

    model = MultiHeadEfficientNet_PE.from_name(num_columns, model_name, num_classes=10)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    state_dict = torch.load(full_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    label_names_rings = ['healthy', 'present_srf', 'present_irf', 'mm1_srf', 'mm3_srf', 'mm6_srf', 'mm1_irf', 'mm3_irf', 'mm6_irf']
    label_names_cols = ['healthy', 'srf', 'irf']
    label_names_cols.extend(['srf_' + str(i) for i in range(num_columns)])
    label_names_cols.extend(['irf_' + str(i) for i in range(num_columns)])

    running_pred_cols = []
    running_pred_rm = []
    running_pred_rw = []
    running_true = []
    slice_numbers = []
    image_names = []

    for i, data in enumerate(dataloader):
        if i % 10 == 0:
            print(i)
        if i > 2:
            break

        inputs, labels, slices = data['images'].to(device).float(), data['labels'].to(device), data['slices'].to(device)
        slice_number, image_name, centers = data['slice_number'], data['image_name'], data['center']

        rel_center = torch.abs(slice_number - centers.squeeze(1)).to(device)

        with torch.no_grad():
            y_bio, y_columns = model(inputs, rel_center)

        y_bio = y_bio[:, :3]
        y_srf = y_columns[:, :, 1]
        y_irf = y_columns[:, :, 2]

        y_bio = torch.sigmoid(y_bio).detach().cpu().numpy()
        y_srf = torch.sigmoid(y_srf).detach().cpu().numpy()
        y_irf = torch.sigmoid(y_irf).detach().cpu().numpy()

        if return_type == 'rm' or return_type == 'a':
            y_srf_rm = u.convert_to_rings(y_srf, slice_number, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
            y_irf_rm = u.convert_to_rings(y_irf, slice_number, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
            l_rm = u.join_labels(y_bio, y_srf_rm, y_irf_rm)
            running_pred_rm.extend(l_rm)

        if return_type == 'rw' or return_type == 'a':
            y_srf_rw = u.convert_to_rings_weighted(y_srf, slice_number, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
            y_irf_rw = u.convert_to_rings_weighted(y_irf, slice_number, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
            l_rw = u.join_labels(y_bio, y_srf_rw, y_irf_rw)
            running_pred_rw.extend(l_rw)

        l = u.join_labels(y_bio, y_srf, y_irf)

        running_pred_cols.extend(l)
        running_true.extend(labels.detach().cpu().numpy())
        slice_numbers.extend(slice_number.numpy())
        image_names.extend(image_name)

    # Maximum rings output
    if return_type == 'rm' or return_type == 'a':
        pred_labels = np.array(running_pred_rm)
        true_labels = np.array(running_true)
        slice_numbers = np.array(slice_numbers)
        image_names = np.array(image_names)

        true_df = pd.DataFrame(true_labels, columns=label_names_rings, index=[image_names, slice_numbers])
        pred_df = pd.DataFrame(pred_labels, columns=label_names_rings, index=[image_names, slice_numbers])

        df = true_df.join(pred_df, lsuffix='_true', rsuffix='_pred')

        df.to_csv(Path(__file__).resolve().parents[0].joinpath(f'results/rm_results_{model_date}.csv'))

    # Weighted rings output
    if return_type == 'rw' or return_type == 'a':
        pred_labels = np.array(running_pred_rw)
        true_labels = np.array(running_true)
        slice_numbers = np.array(slice_numbers)
        image_names = np.array(image_names)

        true_df = pd.DataFrame(true_labels, columns=label_names_rings, index=[image_names, slice_numbers])
        pred_df = pd.DataFrame(pred_labels, columns=label_names_rings, index=[image_names, slice_numbers])

        df = true_df.join(pred_df, lsuffix='_true', rsuffix='_pred')

        df.to_csv(Path(__file__).resolve().parents[0].joinpath(f'results/rw_results_{model_date}.csv'))

    if return_type == 'c' or return_type == 'a':
        pred_labels = np.array(running_pred_cols)
        slice_numbers = np.array(slice_numbers)
        image_names = np.array(image_names)

        df = pd.DataFrame(pred_labels, columns=label_names_cols, index=[image_names, slice_numbers])

        df.to_csv(Path(__file__).resolve().parents[0].joinpath(f'results/column_results_{model_date}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-number',
                        default=4,
                        type=int,
                        help='Model. It accepts int referring to the number of model, i.e. 0 for efficientnet-b0')

    parser.add_argument('-b', '--batch-size',
                        default=64,
                        type=int,
                        help='Size of the batch')

    parser.add_argument('-p', '--model-path',
                        default='pretrained_models/model_b4_maxavg.pth',
                        type=str,
                        help='Model path')

    parser.add_argument('-u', '--ubelix',
                        default=1,
                        type=int,
                        help='Running on ubelix (0 is no)')

    parser.add_argument('-c', '--clahe',
                        default=1,
                        type=int,
                        help='Use CLAHE (0 is no)')

    parser.add_argument('-cols', '--num-columns',
                        default=16,
                        type=int,
                        help='Number of columns')

    parser.add_argument('-r', '--return-type',
                        default='a',
                        type=str,
                        help='Return columns (c), rings maximum (rm), rings weighted (rw) or all (a)')


    FLAGS, unparsed = parser.parse_known_args()
    model_path = FLAGS.model_path
    model_number = FLAGS.model_number
    num_columns = FLAGS.num_columns
    use_clahe = FLAGS.clahe
    return_type = FLAGS.return_type

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[2].joinpath('Datasets', 'Location')
    else:
        data_path = Path(__file__).parents[0].joinpath('Datasets', 'Location')

    generate_labels()
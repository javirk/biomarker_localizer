import torch
from libs.data_retriever_encoding import OCTSliceMaskDataset
import torchvision.transforms as transforms
from pathlib import Path
import libs.utils as u
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_segmentation():
    N_PIXEL_H = 496
    N_PIXEL_W = 512
    num_columns = 16
    pix_col = N_PIXEL_W / num_columns
    thresholds = {'irf': 0.504, 'srf': 0.377}
    ring_diameters = [1, 3, 6]

    cols_range = 513
    # rows_range = 496

    cols = list(range(0, cols_range, 512 // num_columns))  # 513 because we want 512 as well.
    # rows = list(range(0, rows_range, 496 // num_columns))

    tifs = data_path.joinpath('extractedTifsCLAHE')
    labels_path = data_path.joinpath('BSCAN')

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset = OCTSliceMaskDataset(mode='all', image_set=tifs,
                                  srf_gt_filename=labels_path.joinpath('srf_location.csv'),
                                  irf_gt_filename=labels_path.joinpath('irf_location.csv'), transform_image=t,
                                  encoding_type='noencoding')

    seg_data = np.load(seg_file, allow_pickle=True).item()

    label_names = ['healthy', 'present_srf', 'present_irf', 'mm1_srf', 'mm3_srf', 'mm6_srf', 'mm1_irf', 'mm3_irf',
                   'mm6_irf']

    # running_pred = []
    running_true = []
    slice_numbers = []
    image_names = []
    running_pred_rw = []
    running_pred_cols = []

    for i, data in tqdm(enumerate(dataset)):
        labels, slice_number, image_name, centers = data['labels'], data['slice_number'], data['image_name'], data[
            'center']
        centers = torch.ones_like(centers) * 24

        bs = 1
        seg = np.expand_dims(seg_data.get(image_name)[slice_number], axis=0)  # 1 x C x W

        # Convert to columns and then to rings
        y_srf = np.zeros((bs, num_columns))
        y_irf = np.zeros((bs, num_columns))

        for j in range(len(cols) - 1):
            y_irf[:, j] = np.max(seg[:, 1, cols[j]:cols[j + 1]], axis=1)
            y_srf[:, j] = np.max(seg[:, 2, cols[j]:cols[j + 1]], axis=1)

        y_bio = np.ones((bs, 3))

        y_srf_rw = u.convert_to_rings_weighted(y_srf, torch.tensor([slice_number]), centers, N_PIXEL_W, N_PIXEL_H,
                                               ring_diameters, pix_col)
        y_irf_rw = u.convert_to_rings_weighted(y_irf, torch.tensor([slice_number]), centers, N_PIXEL_W, N_PIXEL_H,
                                               ring_diameters, pix_col)
        l_rw = u.join_labels(y_bio, y_srf_rw, y_irf_rw)
        l = u.join_labels(y_bio, y_srf, y_irf)

        running_pred_rw.extend(l_rw)
        running_pred_cols.extend(l)
        running_true.extend(labels.unsqueeze(0).cpu().numpy())
        slice_numbers.append(slice_number)
        image_names.append(image_name)

    pred_labels = np.array(running_pred_rw)
    true_labels = np.array(running_true)
    slice_numbers = np.array(slice_numbers)
    image_names = np.array(image_names)

    true_df = pd.DataFrame(true_labels, columns=label_names, index=[image_names, slice_numbers])
    pred_df = pd.DataFrame(pred_labels, columns=label_names, index=[image_names, slice_numbers])

    df = true_df.join(pred_df, lsuffix='_true', rsuffix='_pred')

    df.to_csv(Path(__file__).resolve().parents[0].joinpath(f'../results/validation_results_seg.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--ubelix',
                        default=1,
                        type=int,
                        help='Running on ubelix (0 is no)')

    parser.add_argument('-s', '--seg-file',
                        default='results/Segmentation_rebuttal/pixcol_seg.npy',
                        type=str,
                        help='Running on ubelix (0 is no)')

    FLAGS, unparsed = parser.parse_known_args()
    seg_file = FLAGS.seg_file

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[3].joinpath('Datasets', 'Location')
    else:
        data_path = Path('/storage/homefs/jg20n729/OCT_Detection/Datasets/Location/')

    convert_segmentation()

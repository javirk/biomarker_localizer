import pandas as pd
import numpy as np
from libs.utils import convert_to_rings, convert_to_rings_weighted

def main(filename, method, labelled_only=True, corrected=True):
    print(filename, method)
    if method == 'weighted':
        conv = convert_to_rings_weighted
        replace_text = 'rw_lo' if labelled_only else 'rw_d'
    elif method == 'max':
        conv = convert_to_rings
        replace_text = 'rm_lo' if labelled_only else 'rm_d'
    else:
        raise ValueError('Wrong method')

    if corrected:
        replace_text += '_c'
    else:
        replace_text += '_u'

    output_filename = filename.replace('column', replace_text)

    filename_gt = 'results/160454 clahe/rw_results_20210221-160454_old.csv'
    if labelled_only:
        centers_file = '../../Datasets/Location/results_center_mod_1.npy'
    else:
        centers_file = '../../Datasets/Location/results_center_mod.npy'

    centers_info = np.load(centers_file, allow_pickle=True).item()

    N_PIXEL_H = 496
    N_PIXEL_W = 512
    num_columns = 16
    pix_col = N_PIXEL_W / num_columns
    ring_diameters = [1, 3, 6]

    df_gt = pd.read_csv(filename_gt, index_col=[0, 1])
    df_gt.index.names = ('image_name', 'bscan')
    df_gt = df_gt[[x for x in df_gt.columns if '_true' in x]]

    df = pd.read_csv(filename, index_col=[0, 1])
    df.index.names = ('image_name', 'bscan')

    srf = df[[x for x in df.columns if 'srf_' in x]].copy()
    irf = df[[x for x in df.columns if 'irf_' in x]].copy()

    df.rename(columns={'healthy': 'healthy_pred',
                       'irf': 'present_irf_pred',
                       'srf': 'present_srf_pred'}, inplace=True)

    srf_data = srf.to_numpy()
    irf_data = irf.to_numpy()

    srf.reset_index(inplace=True)
    irf.reset_index(inplace=True)

    if corrected:
        centers = [centers_info[x] for x in srf.image_name]
    else:
        centers = [24 if x != -1 else x for x in srf.image_name]

    srf_rings = conv(srf_data, srf.bscan, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
    irf_rings = conv(irf_data, irf.bscan, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)

    srf_rings = pd.DataFrame(data=srf_rings, columns=['mm1_srf_pred', 'mm3_srf_pred', 'mm6_srf_pred'], index=[srf.image_name, srf.bscan])
    irf_rings = pd.DataFrame(data=irf_rings, columns=['mm1_irf_pred', 'mm3_irf_pred', 'mm6_irf_pred'], index=[irf.image_name, irf.bscan])

    total = srf_rings.merge(irf_rings, how='inner', left_index=True, right_index=True)
    # total = total.merge(df[['healthy_pred', 'present_srf_pred', 'present_irf_pred']], how='inner', left_index=True, right_index=True)
    total = total.merge(df_gt, how='inner', left_index=True, right_index=True)

    total.to_csv(output_filename)

if __name__ == '__main__':
    # main('results/PE 105322 clahe/column_results_20210225-105322.csv', 'weighted', corrected=True)
    # main('../Grad-Cam/results/column_gc_results_b4.csv', 'weighted', corrected=True)
    main('../../Datasets/Segmentation/segmentation_columns.csv', 'weighted', labelled_only=False, corrected=False)
    # main('results/160454 clahe/column_results_20210221-160454.csv', 'max')
    # main('results/160552 clahe/column_results_20210221-160552.csv', 'weighted')
    # main('results/160552 clahe/column_results_20210221-160552.csv', 'max')
    # main('results/160740 clahe/column_results_20210221-160740.csv', 'weighted')
    # main('results/160740 clahe/column_results_20210221-160740.csv', 'max')
    # main('results/160813 clahe/column_results_20210221-160813.csv', 'weighted')
    # main('results/160813 clahe/column_results_20210221-160813.csv', 'max')
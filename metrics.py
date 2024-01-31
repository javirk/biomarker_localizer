import libs.utils as u
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os
import numpy as np
import seaborn as sns


def calculate_roc_pr(filename, plot=False, output_folder=None):
    if plot:
        if 'rw' in filename:
            output_folder += '/RW/'
        else:
            output_folder += '/RM/'

        os.makedirs(f'{output_folder}', exist_ok=True)
        os.makedirs(f'{output_folder}/ROC/', exist_ok=True)
        os.makedirs(f'{output_folder}/Precision_Recall/', exist_ok=True)

    labels = pd.read_csv(filename, index_col=[0, 1])
    headers = labels.columns
    labels.reset_index(inplace=True)

    label_names = [x[:-5] for x in headers]
    label_names = list(set(label_names))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    ap = dict()
    for name in label_names:
        true_data, pred_data = u.remove_outer_slices(labels, name)

        fpr[name], tpr[name], roc_auc[name] = metrics.roc_curve(true_data, pred_data)
        roc_auc[name] = metrics.auc(fpr[name], tpr[name])
        precision[name], recall[name], _ = metrics.precision_recall_curve(true_data, pred_data)
        ap[name] = metrics.average_precision_score(true_data, pred_data)
        if plot:
            r = name.split('_')[0][-1]
            try:
                b = name.split('_')[1].upper()
                text = f'{b} {r}mm'
            except IndexError:
                text = name

            title_roc = f'ROC {text}'
            subtitle_roc = f'ROC curve (area = %0.2f) {text}' % roc_auc[name]
            plot_filename = f'{output_folder}/ROC/roc_{name}.svg'
            u.plot_roc(fpr[name], tpr[name], roc_auc[name], name, title_roc, plot_filename, subtitle_roc)

            title_ap = f'Precision - Recall {text}'
            subtitle_ap = f'AP = %0.2f {text}' % ap[name]
            plot_filename = f'{output_folder}/Precision_Recall/ap_{name}.svg'
            u.plot_precision_recall(precision[name], recall[name], ap[name], name, title_ap, plot_filename, subtitle_ap)

    return fpr, tpr, roc_auc, precision, recall, ap

def plot_roc_pr(filename, filename_base, plot=False, output_folder=None):
    if plot:
        if 'rw' in filename:
            output_folder += '/RW/'
        else:
            output_folder += '/RM/'

        os.makedirs(f'{output_folder}', exist_ok=True)
        os.makedirs(f'{output_folder}/ROC/', exist_ok=True)
        os.makedirs(f'{output_folder}/Precision_Recall/', exist_ok=True)

    pred_baseline = pd.read_csv(filename_base,index_col=0)
    pred_baseline.set_index(['image_name', 'bscan'], inplace=True)
    pred_baseline.columns = [x + '_base' for x in pred_baseline.columns]
    pred_baseline.reset_index(inplace=True)

    labels = pd.read_csv(filename, index_col=[0, 1])
    headers = labels.columns
    labels.reset_index(inplace=True)
    y = labels.merge(pred_baseline, left_on=['level_0', 'level_1'], right_on=['image_name', 'bscan'], how='inner')

    label_names = [x[:-5] for x in headers]
    label_names = list(set(label_names))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    ap = dict()
    fpr_base = dict()
    tpr_base = dict()
    roc_auc_base = dict()
    precision_base = dict()
    recall_base = dict()
    ap_base = dict()
    lw = 1.5
    # sns.set_theme(style="ticks")
    # sns.despine()
    # sns.color_palette("Paired")
    fig, ax = plt.subplots(2,2, figsize=(12, 10), tight_layout=True)
    for i, b in enumerate(['irf', 'srf']):
        for name in label_names:
            if b not in name or 'present' in name or 'healthy' in name:
                continue
            r = name.split('_')[0][-1] + 'mm'
            true_data, pred_data = u.remove_outer_slices(y, name)
            fpr[r], tpr[r], roc_auc[r] = metrics.roc_curve(true_data, pred_data)
            roc_auc[r] = metrics.auc(fpr[r], tpr[r])
            precision[r], recall[r], _ = metrics.precision_recall_curve(true_data, pred_data)
            ap[r] = metrics.average_precision_score(true_data, pred_data)

            true_data = y[name + '_true']
            base_data = y[name + '_base']

            fpr_base[r], tpr_base[r], roc_auc_base[r] = metrics.roc_curve(true_data, base_data)
            roc_auc_base[r] = metrics.auc(fpr_base[r], tpr_base[r])
            precision_base[r], recall_base[r], _ = metrics.precision_recall_curve(true_data, base_data)
            ap_base[r] = metrics.average_precision_score(true_data, base_data)

        plt.figure()
        for r in sorted(fpr.keys()):
            ax[0, i].plot(fpr[r], tpr[r], lw=lw, label=f'Ours ROC {b.upper()} {r} (area = %0.2f)' % roc_auc[r])
            ax[0, i].plot(fpr_base[r], tpr_base[r], linestyle='-.', color=ax[0, i].lines[-1].get_color(), lw=lw, label=f'PartConvs ROC {b.upper()} {r} (area = %0.2f)' % roc_auc_base[r])
        sns.lineplot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', ax=ax[0, i])
        ax[0, i].set_xlim([0.0, 1.0])
        ax[0, i].set_ylim([0.0, 1.05])
        ax[0, i].set_xlabel('False Positive Rate', fontsize=14)
        ax[0, i].set_ylabel('True Positive Rate', fontsize=14)
        ax[0, i].set_title(f'ROC {b.upper()}')
        ax[0, i].legend(loc="lower right")
        ax[0, i].grid(visible=True)
        # plt.savefig(f'{output_folder}/ROC/roc_{b}.pdf')
        # plt.close()

        plt.figure()
        for r in sorted(fpr.keys()):
            ax[1, i].plot(recall[r], precision[r], lw=lw, label=f'Ours AP {b.upper()} {r}= %0.2f' % ap[r])
            ax[1, i].plot(recall_base[r], precision_base[r], linestyle='-.', color=ax[1, i].lines[-1].get_color(), lw=lw, label=f'PartConvs AP {b.upper()} {r}= %0.2f' % ap_base[r])
        ax[1, i].set_xlim([0.0, 1.0])
        ax[1, i].set_ylim([0.0, 1.05])
        ax[1, i].set_xlabel('Recall', fontsize=14)
        ax[1, i].set_ylabel('Precision', fontsize=14)
        ax[1, i].set_title(f'Precision Recall {b.upper()}')
        ax[1, i].legend(loc="lower right")
        ax[1, i].grid()
        # ax[1, i].grid(visible=True)
        # plt.savefig(f'{output_folder}/Precision_Recall/ap_{b}.pdf')
        # plt.close()
    fig.savefig(f'{output_folder}/all_plots.png')


def output_metrics(filename, output_file=None):
    if output_file is None:
        output_file = f'results/ap_{filename.split(".")[0]}.csv'
    labels = pd.read_csv(filename, index_col=[0, 1])
    headers = labels.columns
    labels.reset_index(inplace=True)

    label_names = [x[:-5] for x in headers]
    label_names = list(set(label_names))

    df = pd.DataFrame(columns=['AUC', 'AP'], index=label_names)
    for name in label_names:
        true_data, pred_data = u.remove_outer_slices_diff(labels, name)
        # true_data = labels[name + '_true']
        # pred_data = labels[name + '_pred']

        roc_auc = metrics.roc_auc_score(true_data, pred_data, average='macro')
        ap = metrics.average_precision_score(true_data, pred_data, average='macro')
        df.loc[name]['AUC'] = roc_auc
        df.loc[name]['AP'] = ap
        # cm = metrics.confusion_matrix(true_data, pred_data)

    df.to_csv(output_file)

def compare_roc(filenames, label_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    ap = dict()

    for file in filenames:
        fpr[file], tpr[file], roc_auc[file], precision[file], recall[file], ap[file] = calculate_roc_pr(file, plot=False)

    fpr = u.swap_keys_dict(fpr)
    tpr = u.swap_keys_dict(tpr)
    roc_auc = u.swap_keys_dict(roc_auc)
    precision = u.swap_keys_dict(precision)
    recall = u.swap_keys_dict(recall)
    ap = u.swap_keys_dict(ap)

    u.plot_roc_group(fpr, tpr, roc_auc, label_names)
    u.plot_ap_group(recall, precision, ap, label_names)

def roc_prob_per_slice(filename, plot_probs=False):
    encoding_type = filename.split('_')[3][:-4]
    loss_function = filename.split('_')[2]

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    n_slices = 49

    label_names = [x[:-5] for x in labels.columns]
    label_names = list(set(label_names))
    auc = {'AUC.' + x: [] for x in label_names}
    probs = {'P.' + x: [] for x in label_names}

    for i_slice in range(n_slices):
        labels_temp = labels.loc[pd.IndexSlice[:, [i_slice]], :].copy()
        for l in label_names:
            true_data = labels_temp[l + '_true']
            pred_data = labels_temp[l + '_pred']
            fpr, tpr, roc = metrics.roc_curve(true_data, pred_data)
            auc['AUC.' + l].append(metrics.auc(fpr, tpr))
            probs['P.' + l].append(pred_data.mean())

    auc.update(probs)

    df = pd.DataFrame.from_dict(auc)
    df = df[['AUC.healthy', 'AUC.present_srf', 'AUC.mm1_srf', 'AUC.mm3_srf', 'AUC.mm6_srf', 'AUC.present_irf', 'AUC.mm1_irf', 'AUC.mm3_irf', 'AUC.mm6_irf',
             'P.healthy', 'P.present_srf', 'P.mm1_srf', 'P.mm3_srf', 'P.mm6_srf', 'P.present_irf', 'P.mm1_irf', 'P.mm3_irf', 'P.mm6_irf']]

    name = filename.split('/')[-1].replace('test_results_', 'auc_prob_')
    df.to_csv('results/AUC/' + name)

    if plot_probs:
        rings = ['present', 'mm1', 'mm3', 'mm6']
        biom = ['srf', 'irf']
        for b in biom:
            fig = plt.figure()
            for r in rings:
                vals = probs['P.' + r + '_' + b]
                plt.plot(range(len(vals)), vals, label=r)
            plt.xlabel('Slice number')
            plt.ylabel('Mean probability')
            plt.legend(loc="upper right")
            plt.title(f'{b} - {encoding_type} - {loss_function}')
            fig.savefig(f'results/Figures/Probabilities/{b}_{encoding_type}_{loss_function}.png')

def metrics_cscan(filename, plot=False, output_folder=None):
    if output_folder is None:
        output_folder = '/'.join(filename.split('/')[:2]) + '/CSCAN'

    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.makedirs(f'{output_folder}/ROC/', exist_ok=True)
    os.makedirs(f'{output_folder}/Precision_Recall/', exist_ok=True)

    true_data = pd.read_csv('../../Datasets/Location/CSCAN/srf_irf_location.csv')
    true_data.columns = [x.split('.')[1] + '_' + x.split('.')[0] + '_true' if '.' in x else x for x in true_data.columns]

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    label_names = [x for x in labels.columns if '_pred' in x and 'healthy' not in x]

    labels = labels[label_names]
    labels = labels.apply(lambda x: u.nonpresent_to_nan(x))

    pred_data_biom = labels.groupby(['image_name'])[label_names].mean()
    # pred_data_biom['healthy_pred'] = labels.groupby(['image_name'])[['healthy_pred']].min()
    total = true_data.merge(pred_data_biom, how='inner', left_on='image_name', right_on=pred_data_biom.index)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    ap = dict()

    for name in label_names:
        name = name[:-5]

        true_data = total[[name + '_true']].astype(int)
        pred_data = total[[name + '_pred']]

        fpr[name], tpr[name], roc_auc[name] = metrics.roc_curve(true_data, pred_data)
        roc_auc[name] = metrics.auc(fpr[name], tpr[name])
        precision[name], recall[name], _ = metrics.precision_recall_curve(true_data, pred_data)
        ap[name] = metrics.average_precision_score(true_data, pred_data)
        print(f'AP {name}: {ap[name]}')
        print(f'AUC {name}: {roc_auc[name]}')
        if plot:
            title_roc = f'ROC {name} - validation'
            plot_filename = f'{output_folder}/ROC/{name}.png'
            u.plot_roc(fpr[name], tpr[name], roc_auc[name], name, title_roc, plot_filename)
            title_ap = f'Precision - Recall {name}'
            plot_filename = f'{output_folder}/Precision_Recall/{name}.png'
            u.plot_precision_recall(precision[name], recall[name], ap[name], name, title_ap, plot_filename)

def metrics_segmentation(filename, plot=False, output_folder=None, use_sample=False):
    if output_folder is None:
        output_folder = '/'.join(filename.split('/')[:2]) + '/Segmentation'

    if use_sample:
        output_folder += '/Sample/'
    else:
        output_folder += '/All/'

    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.makedirs(f'{output_folder}/ROC/', exist_ok=True)
    os.makedirs(f'{output_folder}/Precision_Recall/', exist_ok=True)

    true_data = pd.read_csv('../../Datasets/Segmentation/segmentation_columns.csv')
    if use_sample:
        with open('../../Datasets/Segmentation/random_sample_reviewed.txt') as f:
            sample = f.read().splitlines()
        true_data = true_data[true_data.image_name.isin(sample)].copy()

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    labels.drop(columns=['healthy', 'srf', 'irf'], inplace=True)
    label_names = labels.columns
    labels.reset_index(inplace=True)

    total = true_data.merge(labels, how='inner', on=['image_name', 'bscan'], suffixes=['_true', '_pred'])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    ap = dict()

    df = pd.DataFrame(columns=['AUC', 'AP'], index=['irf', 'srf'])

    for name in label_names:
        true_data = total[[name + '_true']].astype(int)
        pred_data = total[[name + '_pred']]

        fpr[name], tpr[name], roc_auc[name] = metrics.roc_curve(true_data, pred_data)
        roc_auc[name] = metrics.auc(fpr[name], tpr[name])
        precision[name], recall[name], _ = metrics.precision_recall_curve(true_data, pred_data)
        ap[name] = metrics.average_precision_score(true_data, pred_data)
        # df.loc[name]['AUC'] = roc_auc[name]
        # df.loc[name]['AP'] = ap[name]
        if plot:
            title_roc = f'ROC {name} - validation'
            plot_filename = f'{output_folder}/ROC/{name}.png'
            u.plot_roc(fpr[name], tpr[name], roc_auc[name], name, title_roc, plot_filename)
            title_ap = f'Precision - Recall {name}'
            plot_filename = f'{output_folder}/Precision_Recall/{name}.png'
            u.plot_precision_recall(precision[name], recall[name], ap[name], name, title_ap, plot_filename)

    for name in ['irf', 'srf']:
        cols = [x for x in label_names if name in x]
        true_data = total[[x + '_true' for x in cols]].astype(int)
        pred_data = total[[x + '_pred' for x in cols]]

        df.loc[name]['AUC'] = metrics.roc_auc_score(true_data, pred_data, average='micro')
        df.loc[name]['AP'] = metrics.average_precision_score(true_data, pred_data, average='micro')

    df.to_csv(output_folder + 'metrics_seg.csv')

def probability_map(filename, output_folder=None, use_sample=False, plot=True):
    from PIL import Image

    lateral_resolution = 0.01172 # mm/px
    column_resolution = 32 # px/column
    thickness_resolution = 0.12 # mm/slice

    if output_folder is None:
        output_folder = '/'.join(filename.split('/')[:2]) + '/Segmentation'

    if use_sample:
        output_folder += '/Sample/'
    else:
        output_folder += '/All/'

    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.makedirs(f'{output_folder}/Probability/', exist_ok=True)

    image_folder =  '../../Datasets/Location/extractedTifs/'
    true_data = pd.read_csv('../../Datasets/Segmentation/segmentation_columns.csv')
    if use_sample:
        with open('../../Datasets/Segmentation/random_sample_reviewed.txt') as f:
            sample = f.read().splitlines()
        true_data = true_data[true_data.image_name.isin(sample)].copy()

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    labels.drop(columns=['healthy', 'srf', 'irf'], inplace=True)
    label_names = labels.columns
    labels.reset_index(inplace=True)

    total = true_data.merge(labels, how='inner', on=['image_name', 'bscan'], suffixes=['_true', '_pred'])
    image_names = set(total['image_name'])
    total.set_index(['bscan'], inplace=True)
    tv = {'srf': [], 'irf': []}
    area = {'srf': [], 'irf': []}
    for name in image_names:
        if plot:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.subplots(2, 3)

        for i, b in enumerate(['srf', 'irf']):
            true_data = total[total['image_name'] == name][[x + '_true' for x in label_names if b in x]]
            pred_data = total[total['image_name'] == name][[x + '_pred' for x in label_names if b in x]]

            true_data.columns = [x[4:-5] for x in true_data.columns]
            pred_data.columns = [x[4:-5] for x in pred_data.columns]
            tv[b].append([u.np_tv_loss(true_data.to_numpy(), 1), u.np_tv_loss(pred_data.to_numpy(), 1)])

            true_area_colslice = true_data.to_numpy().sum()
            pred_area_colslice = (pred_data.to_numpy() > 0.8).sum()

            true_area = true_area_colslice * lateral_resolution * column_resolution * thickness_resolution
            pred_area = pred_area_colslice * lateral_resolution * column_resolution * thickness_resolution
            area[b].append([true_area, pred_area])

            if plot:
                sns.heatmap(true_data, ax=ax[i, 0], vmin=0, vmax=1, cmap='rocket')
                sns.heatmap(pred_data, ax=ax[i, 1], vmin=0, vmax=1, cmap='rocket')
                ax[i, 0].title.set_text(f'{b.upper()} GT')
                ax[i, 1].title.set_text(f'{b.upper()} pred')

        if plot:
            im = np.array(Image.open(image_folder + f'{name}/img_24.jpg')) / 255.
            ax[0, 2].imshow(im, cmap='gray')

            fig.savefig(f'{output_folder}/Probability/{name}.png')
            plt.close()

    tv_np = np.concatenate((np.array(tv['srf']), np.array(tv['irf'])), axis=-1)
    area_np = np.concatenate((np.array(area['srf']), np.array(area['irf'])), axis=-1)

    df = pd.DataFrame(tv_np, index=image_names, columns=['SRF TV GT', 'SRF TV Pred' ,'IRF TV GT', 'IRF TV Pred'])
    df.to_csv(output_folder + '/total_variation.csv')

    df = pd.DataFrame(area_np, index=image_names, columns=['SRF AREA GT', 'SRF AREA Pred' ,'IRF AREA GT', 'IRF AREA Pred'])
    df.to_csv(output_folder + '/area.csv')


def area_gt_segmentation(filename, output_folder=None, use_sample=False):
    lateral_resolution = 0.01172 # mm/px
    column_resolution = 32 # px/column
    thickness_resolution = 0.12 # mm/slice

    if output_folder is None:
        output_folder = '/'.join(filename.split('/')[:2]) + '/Segmentation'

    if use_sample:
        output_folder += '/Sample/'
    else:
        output_folder += '/All/'

    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.makedirs(f'{output_folder}/Probability/', exist_ok=True)

    image_folder =  '../../Datasets/Location/extractedTifs/'
    true_data = pd.read_csv('../../Datasets/areas_gt_segmentation.csv', index_col=0)
    if use_sample:
        with open('../../Datasets/Segmentation/random_sample_reviewed.txt') as f:
            sample = f.read().splitlines()
        true_data = true_data[true_data.image_name.isin(sample)].copy()

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    labels.drop(columns=['healthy', 'srf', 'irf'], inplace=True)
    label_names = labels.columns
    labels.reset_index(inplace=True)

    total = true_data.merge(labels, how='inner', on=['image_name'], suffixes=['_true', '_pred'])
    image_names = set(total['image_name'])
    labels.set_index(['bscan'], inplace=True)
    area = {'srf': [], 'irf': []}
    for name in image_names:
        for i, b in enumerate(['srf', 'irf']):
            true_area = total[total['image_name'] == name][[b + '_area']].iloc[0].item()
            pred_data = total[total['image_name'] == name][[x for x in label_names if b in x]]
            # true_area = true_data[true_data['image_name'] == name][b]

            pred_data.columns = [x[4:-5] for x in pred_data.columns]

            pred_area_colslice = (pred_data.to_numpy() > 0.8).sum()

            pred_area = pred_area_colslice * lateral_resolution * column_resolution * thickness_resolution
            area[b].append([true_area, pred_area])

    area_np = np.concatenate((np.array(area['srf']), np.array(area['irf'])), axis=-1)

    df = pd.DataFrame(area_np, index=image_names, columns=['SRF AREA GT', 'SRF AREA Pred' ,'IRF AREA GT', 'IRF AREA Pred'])
    df.to_csv(output_folder + '/area_fullsegmentation.csv')



def metrics_segmentation_all(filename, output_folder=None, use_sample=False):
    if output_folder is None:
        output_folder = '/'.join(filename.split('/')[:2]) + '/Segmentation'

    if use_sample:
        output_folder += '/Sample/'
    else:
        output_folder += '/All/'

    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.makedirs(f'{output_folder}/ROC/', exist_ok=True)
    os.makedirs(f'{output_folder}/Precision_Recall/', exist_ok=True)

    true_data = pd.read_csv('../../Datasets/Segmentation/segmentation_columns.csv')
    if use_sample:
        with open('../../Datasets/Segmentation/random_sample_reviewed.txt') as f:
            sample = f.read().splitlines()
        true_data = true_data[true_data.image_name.isin(sample)].copy()

    labels = pd.read_csv(filename, index_col=[0, 1])
    labels.index.names = ('image_name', 'bscan')
    labels.drop(columns=['healthy', 'srf', 'irf'], inplace=True)
    label_names = labels.columns
    labels.reset_index(inplace=True)

    total = true_data.merge(labels, how='inner', on=['image_name', 'bscan'], suffixes=['_true', '_pred'])

    name_prev = ''
    plt.figure()
    for name in label_names:
        if name[:3] != name_prev:
            plt.savefig(f'{output_folder}/ROC/all_{name_prev}.png')
            plt.close()
            plt.figure(figsize=(16,9), dpi=300)
            lw = 2
            plt.title(f'ROC {name[:3]} - segmentation')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

        true_data = total[[name + '_true']].astype(int)
        pred_data = total[[name + '_pred']]

        fpr, tpr, roc_auc = metrics.roc_curve(true_data, pred_data)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=lw, label=f'ROC curve (area = %0.2f) {name}' % roc_auc)
        plt.legend(loc="lower right")

        name_prev = name[:3]

    plt.savefig(f'{output_folder}/ROC/all_{name[:3]}.png')

def metrics_lambda_rings(filenames, lambda_values, plot=False):
    roc_auc = []
    ap = []
    for filename, lambda_value in zip(filenames, lambda_values):
        labels = pd.read_csv(filename, index_col=[0, 1])
        labels.drop(columns=['present_irf_pred', 'present_srf_pred', 'healthy_pred'], inplace=True)
        headers = labels.columns
        labels.reset_index(inplace=True)

        classes = [x[:-5] for x in headers if '_pred' in x]

        true_data = labels[[x + '_true' for x in classes]]
        pred_data = labels[[x + '_pred' for x in classes]]

        ap.append(metrics.average_precision_score(true_data, pred_data, average='micro') * 100)
        roc_auc.append(metrics.roc_auc_score(true_data, pred_data, average='micro') * 100)

    if plot:
        plt.plot(lambda_values, ap, marker='o')
        plt.ylim([80, 90])
        plt.ylabel('mAP')
        plt.xlabel(r'$\lambda$')
        plt.legend(loc="lower right")
        plt.savefig('results/Figures/ap_lambda_rings.png')
        plt.close()

        plt.plot(lambda_values, roc_auc, marker='o')
        plt.ylim([90, 100])
        plt.ylabel('ROC AUC')
        plt.xlabel(r'$\lambda$')
        plt.legend(loc="lower right")
        plt.savefig('results/Figures/roc_lambda_rings.png')
        plt.close()

    return ap, roc_auc

def metrics_lambda_columns(filenames, lambda_values, use_sample=True, plot=False):
    roc_auc = []
    ap = []

    true_data_all = pd.read_csv('../../Datasets/Segmentation/segmentation_columns.csv')
    if use_sample:
        with open('../../Datasets/Segmentation/random_sample_reviewed.txt') as f:
            sample = f.read().splitlines()
        true_data_all = true_data_all[true_data_all.image_name.isin(sample)].copy()

    for filename, lambda_value in zip(filenames, lambda_values):
        labels = pd.read_csv(filename, index_col=[0, 1])
        labels.index.names = ('image_name', 'bscan')
        labels.drop(columns=['healthy', 'srf', 'irf'], inplace=True)
        classes = labels.columns
        labels.reset_index(inplace=True)

        total = true_data_all.merge(labels, how='inner', on=['image_name', 'bscan'], suffixes=['_true', '_pred'])

        true_data = total[[x + '_true' for x in classes]]
        pred_data = total[[x + '_pred' for x in classes]]

        ap.append(metrics.average_precision_score(true_data, pred_data, average='micro') * 100)
        roc_auc.append(metrics.roc_auc_score(true_data, pred_data, average='micro') * 100)

    if plot:
        plt.plot(lambda_values, ap, marker='o')
        plt.ylim([80, 90])
        plt.ylabel('mAP')
        plt.xlabel(r'$\lambda$')
        plt.legend(loc="lower right")
        plt.savefig('results/Figures/ap_lambda_cols.png')
        plt.close()

        plt.plot(lambda_values, roc_auc, marker='o')
        plt.ylim([90, 100])
        plt.ylabel('ROC AUC')
        plt.xlabel(r'$\lambda$')
        plt.legend(loc="lower right")
        plt.savefig('results/Figures/roc_lambda_cols.png')
        plt.close()

    return ap, roc_auc

def metrics_lambda(filenames_rings, filenames_columns, lambda_values):
    ap_c, roc_c = metrics_lambda_columns(filenames_columns, lambda_values, use_sample=True, plot=False)
    ap_r, roc_r = metrics_lambda_rings(filenames_rings, lambda_values, plot=False)

    order = np.argsort(lambda_values)
    lambda_values = np.array(lambda_values)[order]
    ap_c = np.array(ap_c)[order]
    ap_r = np.array(ap_r)[order]
    roc_c = np.array(roc_c)[order]
    roc_r = np.array(roc_r)[order]

    plt.figure(figsize=(5, 5*3/4))
    plt.plot(lambda_values, ap_c, marker='o', label='Columns')
    plt.plot(lambda_values, ap_r, marker='o', label='Rings')

    plt.ylim([70, 90])
    plt.ylabel('mAP')
    plt.xlabel(r'$\lambda$')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc="lower right")
    plt.savefig('results/Figures/ap_lambda.png')
    plt.savefig('results/Figures/ap_lambda.eps', format='eps')
    plt.close()

    plt.figure(figsize=(5, 5*3/4))
    plt.plot(lambda_values, roc_c, marker='o', label='Columns')
    plt.plot(lambda_values, roc_r, marker='o', label='Rings')

    plt.ylim([80, 100])
    plt.ylabel('ROC AUC')
    plt.xlabel(r'$\lambda$')

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc="lower right")
    plt.savefig('results/Figures/roc_lambda.png')
    plt.savefig('results/Figures/roc_lambda.eps', format='eps')
    plt.close()


if __name__ == '__main__':
    # probability_map('results/155503 clahe - SEG/column_results_20210301-155503.csv', use_sample=True, plot=False)
    # area_gt_segmentation('results/155503 clahe - SEG/column_results_20210301-155503.csv', use_sample=True)

    # calculate_roc_pr('results/155503 clahe/rw_results_20210301-155503.csv', plot=True, output_folder='results/155503 clahe/')
    # calculate_roc_pr('results/full_supervision/rw_results_20230924-160656.csv', plot=True, output_folder='results/full_supervision')
    # plot_roc_pr('results/155503 clahe/rw_results_20210301-155503.csv',
    #             '../Mask/results_pconv/mask_probs.csv',
    #             plot=True, output_folder='results/155503 clahe/')

    # output_metrics('results/mscam_rebuttal/rw_mscam_results_cam2.csv',
    #                output_file='results/mscam_rebuttal/metrics_mscam.csv')

    output_metrics('results/155503 clahe/rw_results_short_20210301-155503.csv',
                   output_file='results/155503 clahe/metrics_short_20210301-155503.csv')

    '''First slide'''
    # filenames = ['results/111853 no clahe/validation_results_20210212-111853.csv',
    #              'results/111853 clahe/validation_results_20210212-111853.csv',
    #              '../No_encoding/results/124200 PE/validation_results_20210203-124200_pe.csv']
    #
    # label_names = ['16 cols',
    #                '16 cols + CLAHE',
    #                'Avg PE (rings)']
    #
    # compare_roc(filenames, label_names)

    '''Second slide'''
    # filenames = ['results/191833 clahe/rw_results_20210217-191833.csv',
    #              'results/191925 clahe/rw_results_20210217-191925.csv',
    #              'results/192007 clahe/rw_results_20210217-192007.csv']
    #
    # label_names = ['0 Dkl',
    #                '0.1 Dkl',
    #                '0.01 Dkl']
    #
    # compare_roc(filenames, label_names)

    # metrics_cscan('results/155503 clahe/rw_results_20210301-155503.csv', plot=True)

    # metrics_segmentation('results/191833 no clahe/column_results_20210217-191833.csv', plot=True, use_sample=True)
    # metrics_segmentation('results/191833 clahe/column_results_20210217-191833.csv', plot=True, use_sample=True)
    # metrics_segmentation('results/192007 no clahe/column_results_20210217-192007.csv', plot=True, use_sample=True)
    # metrics_segmentation('results/192007 clahe/column_results_20210217-192007.csv', plot=True, use_sample=True)
    # metrics_segmentation('results/191925 no clahe/column_results_20210217-191925.csv', plot=True, use_sample=True)
    # metrics_segmentation('results/155503 clahe/column_results_20210301-155503.csv', plot=True, use_sample=True)

    # Metrics using segmentation (all together in one plot)
    # metrics_segmentation_all('results/160454 clahe/column_results_20210221-160454.csv', use_sample=True)

    # Metrics depending on lambda
    lambdas = [0.2, 0.0, 0.5, 0.01, 0.05, 0.1, 0.8]
    # For rings
    filenames_rings = ['results/PE 105322 clahe/rw_results_20210225-105322.csv',
                       'results/PE 113121 clahe/rw_results_20210225-113121.csv',
                       'results/PE 130032 clahe/rw_results_20210225-130032.csv',
                       'results/PE 130542 clahe/rw_results_20210225-130542.csv',
                       'results/PE 140755 clahe/rw_results_20210225-140755.csv',
                       'results/PE 124405 clahe/rw_results_20210225-124405.csv',
                       'results/PE 165913 clahe/rw_results_20210225-165913.csv']
    # metrics_lambda_rings(filenames_rings, lambdas)

    # for f in filenames_rings:
    #     name = f.split('/')[-1]
    #     calculate_roc_pr(f, plot=True, output_folder='/'.join(f.split('/')[:2]))
    #     output_metrics(f, output_file='/'.join(f.split('/')[:2]) + '/' + name.replace('rw_results', 'metrics'))

    # For columns
    filenames_columns = ['results/PE 105322 clahe/column_results_20210225-105322.csv',
                         'results/PE 113121 clahe/column_results_20210225-113121.csv',
                         'results/PE 130032 clahe/column_results_20210225-130032.csv',
                         'results/PE 130542 clahe/column_results_20210225-130542.csv',
                         'results/PE 140755 clahe/column_results_20210225-140755.csv',
                         'results/PE 124405 clahe/column_results_20210225-124405.csv',
                         'results/PE 165913 clahe/column_results_20210225-165913.csv']
    # metrics_lambda_columns(filenames_columns, lambdas)

    lambdas = [0.1, 0.01, 0.0, 0.5, 0.2, 0.05, 0.8]
    # For rings
    filenames_rings = ['results/191925 clahe/rw_results_20210217-191925.csv',
                       'results/192007 clahe/rw_results_20210217-192007.csv',
                       'results/191833 clahe/rw_results_20210217-191833.csv',
                       'results/160454 clahe/rw_results_20210221-160454.csv',
                       'results/160552 clahe/rw_results_20210221-160552.csv',
                       'results/160740 clahe/rw_results_20210221-160740.csv',
                       'results/160813 clahe/rw_results_20210221-160813.csv']

    filenames_columns = ['results/191925 clahe/column_results_20210217-191925.csv',
                       'results/192007 clahe/column_results_20210217-192007.csv',
                       'results/191833 clahe/column_results_20210217-191833.csv',
                       'results/160454 clahe/column_results_20210221-160454.csv',
                       'results/160552 clahe/column_results_20210221-160552.csv',
                       'results/160740 clahe/column_results_20210221-160740.csv',
                       'results/160813 clahe/column_results_20210221-160813.csv']
    # metrics_lambda(filenames_rings, filenames_columns, lambdas)
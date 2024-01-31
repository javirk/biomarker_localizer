import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever_encoding import OCTSliceMaskDataset, SegmentationColumnDataset, Resize
import torchvision.transforms as transforms
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from libs.models import MultiHeadEfficientNet


def main():
    N_PIXEL_H = 496
    N_PIXEL_W = 512
    pix_col = N_PIXEL_W / num_columns
    ring_diameters = [1, 3, 6]

    model_name = f'efficientnet-b{model_number}'
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = Path(__file__).resolve().parents[0].joinpath('runs/TL_{}'.format(current_time))
    pretrained_model_path = Path(__file__).resolve().parents[0].joinpath('pretrained_models', 'model_b4_maxavg.pth')

    with open(Path(__file__).resolve().parents[0].joinpath('runs', 'info.txt'), 'a') as file:
        file.write(f'{current_time}: {N_EPOCHS} epochs, LR: {LR} Columns: {num_columns}\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)

    tifs = data_path.joinpath('Location', 'extractedTifs')
    labels_path = data_path.joinpath('Location', 'BSCAN')
    folds_path = labels_path.joinpath('FOLDS/5-fold')  # Validation in one fold to make it faster
    segmentations_path = data_path.joinpath('Segmentation', 'segmentation_columns.csv')
    sample_path = data_path.joinpath('Segmentation', 'random_sample_reviewed.txt')

    minibatch_size = u.max_minibatch_size(model_name)

    if BATCH_SIZE > minibatch_size:
        minibatch_update = BATCH_SIZE // minibatch_size
    else:
        minibatch_size = BATCH_SIZE
        minibatch_update = 1

    t = transforms.Compose([Resize(512), transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    t_val = transforms.Compose([Resize(512), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    trainset = SegmentationColumnDataset(image_root=tifs, label_root=segmentations_path, sample_file=sample_path,
                                         transform_image=t)

    valset = OCTSliceMaskDataset(mode='validation', image_set=tifs,
                                 srf_gt_filename=labels_path.joinpath('srf_location.csv'),
                                 irf_gt_filename=labels_path.joinpath('irf_location.csv'), transform_image=t_val,
                                 fold=0, folds_dir=folds_path, encoding_type='noencoding')

    trainset_size = len(trainset)
    valset_size = len(valset)

    writing_freq_train = trainset_size // (writing_per_epoch * minibatch_size)
    writing_freq_val = valset_size // minibatch_size

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=minibatch_size, shuffle=False, num_workers=0)
    # mask_generator = u.generate_mask(ring_diameters, 49, N_PIXEL_H, N_PIXEL_W, )

    model = MultiHeadEfficientNet.from_name(num_columns, model_name, num_classes=3)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6, momentum=0.1)
    lmbda = lambda epoch: 0.99
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    main_criterion = nn.BCEWithLogitsLoss()

    best_roc = 0.0
    for epoch in range(N_EPOCHS):
        for phase in ['train', 'validation']:  # TRAIN
            running_loss = 0.0
            running_pred = []
            running_true = []

            if phase == 'train':
                model.train()
                loader = trainloader
                writing_freq = writing_freq_train
                i_train = 0
            elif phase == 'validation':
                model.eval()
                loader = valloader
                writing_freq = writing_freq_val

            for i, data in enumerate(loader):
                inputs, labels, slices = data['images'].to(device).float(), data['labels'].to(device).float(), data['slice_number']
                centers = data['center']

                with torch.set_grad_enabled(phase == 'train'):
                    y_bio, y_columns = model(inputs)
                    # y_bio [batch_size, 10]
                    # y_columns [batch_size, num_columns, 10]
                    y_columns = y_columns[..., :3]

                    # Labels come in the form [batch_size, 3*num_columns]
                    # Convert y_columns to [batch_size, 3*num_columns]
                    y_columns_reshape = y_columns.flatten(start_dim=1)


                    if phase == 'train':
                        loss = main_criterion(y_columns_reshape, labels)
                        i_train = i
                        loss.backward()
                        if i % minibatch_update == (minibatch_update - 1):
                            optimizer.step()
                            optimizer.zero_grad()
                        running_loss += loss.item()

                if phase == 'validation':
                    y_bio = torch.sigmoid(y_bio).detach().cpu().numpy()
                    y_srf = y_columns[:, :, 1]
                    y_irf = y_columns[:, :, 2]
                    y_srf = torch.sigmoid(y_srf).detach().cpu().numpy()
                    y_irf = torch.sigmoid(y_irf).detach().cpu().numpy()

                    y_srf = u.convert_to_rings(y_srf, slices, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)
                    y_irf = u.convert_to_rings(y_irf, slices, centers, N_PIXEL_W, N_PIXEL_H, ring_diameters, pix_col)

                    # Healthy, SRF Present, IRF Present, SRF1, SRF3, SRF6, IRF1, IRF3, IRF6
                    l = u.join_labels(y_bio[:, :3], y_srf, y_irf)
                    running_pred.append(l)
                    running_true.append(labels.detach().cpu().numpy())
                else:
                    # During training there is only information of the biomarkers, not the rings
                    y_columns_reshape = torch.sigmoid(y_columns_reshape).detach().cpu().numpy()
                    running_pred.append(y_columns_reshape)
                    running_true.append(labels.detach().cpu().numpy())

                if i % writing_freq == (writing_freq - 1):
                    n_epoch = epoch * trainset_size // minibatch_size + i_train + 1
                    epoch_loss = running_loss / (writing_freq * minibatch_size)
                    dict_metrics = u.calculate_metrics_sigmoid(running_true, running_pred)
                    epoch_rocauc = dict_metrics['ROC AUC']
                    print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_rocauc}')
                    dict_metrics['Loss'] = epoch_loss
                    u.write_to_tb(writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                    running_pred = []
                    running_true = []
                    running_loss = 0.0

                    if phase == 'validation' and epoch_rocauc > best_roc:
                        best_roc = epoch_rocauc
                        # best_model_wts = copy.deepcopy(model.state_dict())

            scheduler.step()
            print(f'Epoch {epoch} finished')
        torch.save(model.state_dict(),
                   Path(__file__).parents[0].joinpath('weights', f'weights_{current_time}_e{epoch}.pth'))


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

    parser.add_argument('-e', '--epochs',
                        default=100,
                        type=int,
                        help='Number of epochs')

    parser.add_argument('-lr', '--learning-rate',
                        default=0.256,
                        type=float,
                        help='Initial learning rate')

    parser.add_argument('-w', '--writing-per-epoch',
                        default=100,
                        type=int,
                        help='Times to write per epoch')

    parser.add_argument('-u', '--ubelix',
                        default=1,
                        type=int,
                        help='Running on ubelix (0 is no)')

    parser.add_argument('-cols', '--num-columns',
                        default=16,
                        type=int,
                        help='Number of columns')

    FLAGS, unparsed = parser.parse_known_args()
    model_number = FLAGS.model_number
    BATCH_SIZE = FLAGS.batch_size
    N_EPOCHS = FLAGS.epochs
    LR = FLAGS.learning_rate
    writing_per_epoch = FLAGS.writing_per_epoch
    num_columns = FLAGS.num_columns

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[2].joinpath('Datasets')
    else:
        data_path = Path(__file__).parents[0].joinpath('Datasets')

    main()

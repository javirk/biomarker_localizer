import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever import OCTHDF5Dataset, Resize
from libs.data_retriever_encoding import OCTSliceMaskDataset, CLAHE
import torchvision.transforms as transforms
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from libs.models import MultiHeadEfficientNet
from libs.custom_loss import MultiHeadLoss


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
        file.write(f'{current_time}: {N_EPOCHS} epochs, LR: {LR}, Lambda: {lambda_value} Columns: {num_columns}\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)

    tifs = data_path.joinpath('Location', 'extractedTifs')
    labels_path = data_path.joinpath('Location', 'BSCAN')
    folds_path = labels_path.joinpath('FOLDS/5-fold')
    train_path = data_path.joinpath('ambulatorium_all_slices.hdf5')

    minibatch_size = u.max_minibatch_size(model_name)

    if BATCH_SIZE > minibatch_size:
        minibatch_update = BATCH_SIZE // minibatch_size
    else:
        minibatch_size = BATCH_SIZE
        minibatch_update = 1

    t = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    t_val = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    trainset = OCTHDF5Dataset(train_path, transform_image=t)

    valset = OCTSliceMaskDataset(mode='validation', image_set=tifs,
                                 srf_gt_filename=labels_path.joinpath('srf_location.csv'),
                                 irf_gt_filename=labels_path.joinpath('irf_location.csv'), transform_image=t_val,
                                 fold=0, folds_dir=folds_path, encoding_type='noencoding')

    trainset_size = len(trainset)
    valset_size = len(valset)

    writing_freq_train = trainset_size // (writing_per_epoch * minibatch_size)
    writing_freq_val = valset_size // minibatch_size

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset, batch_size=minibatch_size, shuffle=False, num_workers=1)
    mask_generator = u.generate_mask(ring_diameters, 49, N_PIXEL_H, N_PIXEL_W, )

    model = MultiHeadEfficientNet.from_name(num_columns, model_name, num_classes=10)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6, momentum=0.1)
    lmbda = lambda epoch: 0.99
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    main_criterion = nn.BCEWithLogitsLoss()
    srf_criterion = MultiHeadLoss(num_columns)
    irf_criterion = MultiHeadLoss(num_columns)

    best_roc = 0.0
    for epoch in range(N_EPOCHS):
        for phase in ['train', 'validation']:
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
                inputs, labels, slices = data['images'].to(device).float(), data['labels'].to(device), data['slices']
                if phase == 'validation':
                    labels_cut = torch.index_select(labels, 1, torch.as_tensor([0, 1, 2]).to(
                        device)).float()  # Healthy, SRF and IRF
                    slices = torch.argmax(slices, dim=2).squeeze(1)
                else:
                    labels_cut = labels[:, :3].float()

                with torch.set_grad_enabled(phase == 'train'):
                    y_bio, y_columns = model(inputs)

                    y_bio = y_bio[:, :3].to(device)
                    y_srf = y_columns[:, :, 1].to(device)
                    y_irf = y_columns[:, :, 2].to(device)

                    loss_biom = main_criterion(y_bio, labels_cut)
                    loss_srf = srf_criterion(y_srf, labels_cut[:, 1])
                    loss_irf = irf_criterion(y_irf, labels_cut[:, 2])

                    loss = loss_biom + lambda_value * (loss_srf + loss_irf)

                    if phase == 'train':
                        i_train = i
                        loss.backward()
                        if i % minibatch_update == (minibatch_update - 1):
                            optimizer.step()
                            optimizer.zero_grad()

                running_loss += loss.item()

                y_bio = torch.sigmoid(y_bio).detach().cpu().numpy()

                if phase == 'validation':
                    y_srf = torch.sigmoid(y_srf).detach().cpu().numpy()
                    y_irf = torch.sigmoid(y_irf).detach().cpu().numpy()

                    y_srf = u.convert_to_rings(y_srf, slices, mask_generator, pix_col)
                    y_irf = u.convert_to_rings(y_irf, slices, mask_generator, pix_col)

                    # Healthy, SRF Present, IRF Present, SRF1, SRF3, SRF6, IRF1, IRF3, IRF6
                    l = u.join_labels(y_bio[:, :3], y_srf, y_irf)
                    running_pred.append(l)
                    running_true.append(labels.detach().cpu().numpy())
                else:
                    # During training there is only information of the biomarkers, not the rings
                    running_pred.append(y_bio)
                    running_true.append(labels_cut.detach().cpu().numpy())

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

    parser.add_argument('-l', '--lambda-loss',
                        default=1.,
                        type=float,
                        help='Lambda value for loss function')

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
    lambda_value = FLAGS.lambda_loss
    num_columns = FLAGS.num_columns

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[2].joinpath('Datasets')
    else:
        data_path = Path(__file__).parents[0].joinpath('Datasets')

    main()

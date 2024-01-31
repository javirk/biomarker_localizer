import torch
import torch.nn as nn
import torch.nn.functional as F


# Taken from https://github.com/usuyama/pytorch-unet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes_classification, n_classes_segmentation,
                 segment=True, bilinear=True, features_cl=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes_classification = n_classes_classification
        self.n_classes_segmentation = n_classes_segmentation
        self.bilinear = bilinear
        self.segment = segment

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes_segmentation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(1024 // factor * 2, n_classes_classification)  # * 2 because avg and max pooling are
        # concatenated
        self.fc_decoder = nn.Linear(64 * 2, n_classes_classification)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        im = self.classifier(x5)
        if self.segment:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            seg = self.outc(x)
            im_decoder = self.classifier_decoder(x)
        else:
            seg = None
            im_decoder = None

        return {'im': im, 'seg': seg, 'im_decoder': im_decoder}

    def classifier(self, x):
        x_avg = self.avgpool(x).flatten(start_dim=1)
        x_max = self.maxpool(x).flatten(start_dim=1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc(x)
        return x

    def classifier_decoder(self, x):
        x_avg = self.avgpool(x).flatten(start_dim=1)
        x_max = self.maxpool(x).flatten(start_dim=1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc_decoder(x)
        return x

    def features_cl(self, x):
        x_avg = self.avgpool(x).flatten(start_dim=1)
        x_max = self.maxpool(x).flatten(start_dim=1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.relu(self.fc_features_hidden(x))
        x = self.fc_features(x)
        return x

    def set_segmentation(self, mode):
        self.segment = mode
        print(f'Segment is now {self.segment}')
        if self.segment:
            self.up1.requires_grad_(True)
            self.up2.requires_grad_(True)
            self.up3.requires_grad_(True)
            self.up4.requires_grad_(True)
            self.outc.requires_grad_(True)
            self.fc.requires_grad_(True)
            self.fc_decoder.requires_grad_(True)
            # self.fc_features.requires_grad_(False)
            # self.fc_features_hidden.requires_grad_(False)
        else:
            self.up1.requires_grad_(False)
            self.up2.requires_grad_(False)
            self.up3.requires_grad_(False)
            self.up4.requires_grad_(False)
            self.outc.requires_grad_(False)
            self.fc.requires_grad_(True)
            # self.fc_features.requires_grad_(True)
            # self.fc_features_hidden.requires_grad_(True)


if __name__ == '__main__':
    from torchsummaryX import summary

    m = UNet(3, 10, 10, bilinear=False)
    m.eval()
    print(summary(m, torch.zeros((1, 3, 512, 512))))

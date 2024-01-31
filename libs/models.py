from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters, get_model_params
import torch
import torch.nn as nn


class CustomEfficientNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        # Average pooling is already created under self._avg_pooling
        out_channels = round_filters(1280, self._global_params)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        self._fc_biomarker = nn.Linear(out_channels * 2, 3)
        self._fc_srf = nn.Linear(out_channels * 2, 10)
        self._fc_irf = nn.Linear(out_channels * 2, 10)
        del self._fc  # Otherwise it collides with the trained model weights

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)
        x = torch.cat((x, x), dim=1)
        x_avg = self._avg_pooling(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling(x[:, x.shape[1] // 2:])

        x = torch.cat((x_avg, x_max), dim=1)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x_biom = self._fc_biomarker(x)
        x_srf = self._fc_srf(x)
        x_irf = self._fc_irf(x)

        return x_biom, x_srf, x_irf


class MultiHeadEfficientNet(EfficientNet):
    def __init__(self, num_columns, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        assert num_columns in [8, 16], 'Number of columns must be either 8 or 16'
        if num_columns == 8:
            kernel_size = (16, 2)
        else:
            kernel_size = (16, 1)

        out_channels = round_filters(1280, self._global_params)
        # Average pooling is already created under self._avg_pooling
        self._max_pooling = nn.AdaptiveMaxPool2d(1)

        self._max_pooling_head = nn.MaxPool2d(kernel_size=kernel_size)
        self._avg_pooling_head = nn.AvgPool2d(kernel_size=kernel_size)
        self._fc_new = nn.Linear(out_channels * 2, self._global_params.num_classes)
        del self._fc

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = torch.cat((x, x), dim=1)
        y_image = self.extract_whole_image(x)

        x_avg = self._avg_pooling_head(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling_head(x[:, x.shape[1] // 2:])
        x = torch.cat((x_avg, x_max), dim=1).squeeze(2)
        x = x.permute(0, 2, 1)

        prev_dim = (x.shape[0], x.shape[1])
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        x = self._dropout(x)
        x = self._fc_new(x)
        y = x.reshape((prev_dim[0], prev_dim[1], x.shape[-1]))

        # y = torch.empty(x.shape[0], x.shape[1] + 1, self._global_params.num_classes)
        # y[:, 0, :] = y_image
        #
        # num_columns = x.shape[1]
        # for i in range(num_columns):
        #     x_head = x[:, i]
        #     x_head = self._dropout(x_head)
        #     x_head = self._fc(x_head)
        #     y[:, i + 1, :] = x_head

        return y_image, y

    def extract_whole_image(self, inputs):
        x = inputs.clone()
        x_avg = self._avg_pooling(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling(x[:, x.shape[1] // 2:])

        x = torch.cat((x_avg, x_max), dim=1)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc_new(x)
        return x
    
    @classmethod
    def from_name(cls, num_columns, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            num_columns (int): Number of columns in the output
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(num_columns, blocks_args, global_params)
        model._change_in_channels(in_channels)

        return model


class MultiHeadEfficientNet_AllClasses(EfficientNet):
    def __init__(self, num_columns, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        assert num_columns in [8, 16], 'Number of columns must be either 8 or 16'
        if num_columns == 8:
            kernel_size = (16, 2)
        else:
            kernel_size = (16, 1)

        out_channels = round_filters(1280, self._global_params)
        # Average pooling is already created under self._avg_pooling
        self._max_pooling = nn.AdaptiveMaxPool2d(1)

        self._max_pooling_head = nn.MaxPool2d(kernel_size=kernel_size)
        self._avg_pooling_head = nn.AvgPool2d(kernel_size=kernel_size)
        self._fc = nn.Linear(out_channels * 2, self._global_params.num_classes)

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = torch.cat((x, x), dim=1)
        y_image = self.extract_whole_image(x)

        x_avg = self._avg_pooling_head(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling_head(x[:, x.shape[1] // 2:])
        x = torch.cat((x_avg, x_max), dim=1).squeeze(2)
        x = x.permute(0, 2, 1)

        prev_dim = (x.shape[0], x.shape[1])
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        x = self._dropout(x)
        x = self._fc(x)
        y = x.reshape((prev_dim[0], prev_dim[1], x.shape[-1]))

        # y = torch.empty(x.shape[0], x.shape[1] + 1, self._global_params.num_classes)
        # y[:, 0, :] = y_image
        #
        # num_columns = x.shape[1]
        # for i in range(num_columns):
        #     x_head = x[:, i]
        #     x_head = self._dropout(x_head)
        #     x_head = self._fc(x_head)
        #     y[:, i + 1, :] = x_head

        return y_image, y

    def extract_whole_image(self, inputs):
        x = inputs.clone()
        x_avg = self._avg_pooling(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling(x[:, x.shape[1] // 2:])

        x = torch.cat((x_avg, x_max), dim=1)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, num_columns, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            num_columns (int): Number of columns in the output
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(num_columns, blocks_args, global_params)
        model._change_in_channels(in_channels)

        return model


if __name__ == '__main__':
    xrand = torch.ones((2, 3, 496, 512))
    model = MultiHeadEfficientNet.from_name(16, 'efficientnet-b4', num_classes=10)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    state_dict = torch.load('../weights/weights_20210211-131909_e4.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    a = model(xrand)

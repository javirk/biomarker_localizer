from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters, get_model_params
import torch
import torch.nn as nn
from libs.positional_encoding import PositionalEncoding


class MultiHeadEfficientNet_PE(EfficientNet):
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
        self._pe = PositionalEncoding(512, n_position=49)
        self._fc_pe = nn.Linear(out_channels * 2 + 512, self._global_params.num_classes)
        del self._fc

    def forward(self, inputs, slice_pos):
        '''
        Forward
        :param inputs:
        :param slice_pos: Slice position relative to center. Positive value, so abs(slice_number-center)
        :return:
        '''
        x = self.extract_features(inputs)
        x = torch.cat((x, x), dim=1)
        with torch.no_grad():
            pe = self._pe(slice_pos)
        y_image = self.extract_whole_image(x, pe)

        x_avg = self._avg_pooling_head(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling_head(x[:, x.shape[1] // 2:])
        x = torch.cat((x_avg, x_max), dim=1).squeeze(2)
        x = x.permute(0, 2, 1)

        prev_dim = (x.shape[0], x.shape[1])
        pe = pe.squeeze(2).squeeze(2).unsqueeze(1)
        pe = pe.repeat(1, x.shape[1], 1)
        x = torch.cat((x, pe), dim=2)
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))

        x = self._dropout(x)
        x = self._fc_pe(x)
        y = x.reshape((prev_dim[0], prev_dim[1], x.shape[-1]))

        return y_image, y

    def extract_whole_image(self, inputs, pe):
        x = inputs.clone()
        x_avg = self._avg_pooling(x[:, :x.shape[1] // 2])
        x_max = self._max_pooling(x[:, x.shape[1] // 2:])

        x = torch.cat((x_avg, x_max, pe), dim=1)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc_pe(x)
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
    nums = torch.tensor([0,1])
    model = MultiHeadEfficientNet_PE.from_name(16, 'efficientnet-b4', num_classes=10)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    state_dict = torch.load('../pretrained_models/model_b4_maxavg.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    a = model(xrand, nums)

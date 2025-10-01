import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    """
    Ideas comes from AutoFormer
    Decompose a time series into trends and seasonal
    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        t_x = x.transpose(1, 2)  # Transpose from [N, L, C] to [N, C, L]
        mean_x = F.avg_pool1d(
            t_x, self.kernel_size, stride=1, padding=self.kernel_size // 2)  # 'SAME' padding equivalent
        mean_x = mean_x.transpose(1, 2)  # Transpose back to [N, L, C]
        return x - mean_x, mean_x


class WPFModel(nn.Module):
    """Models for Wind Power Prediction"""

    def __init__(self, settings):
        super(WPFModel, self).__init__()
        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]

        DECOMP = 18
        self.decomp = SeriesDecomp(DECOMP)

        a = float(1 / self.input_len)
        # season
        self.Linear_Seasonal = nn.Linear(self.input_len, self.output_len)
        x1 = a * torch.ones([self.input_len, self.output_len], dtype=torch.float32)
        self.Linear_Seasonal.weight = nn.Parameter(x1)

        # trend
        self.Linear_Trend = nn.Linear(self.input_len, self.output_len)
        y1 = a * torch.ones([self.input_len, self.output_len], dtype=torch.float32)
        self.Linear_Trend.weight = nn.Parameter(y1)

    def forward(self, batch_x):
        """
        :param batch_x: [N, L, C]  C=134(turbines)
        :return: prediction output
        """
        seasonal_init, trend_init = self.decomp(batch_x)
        seasonal_init = seasonal_init.transpose(1, 2)  # Transpose from [N, L, C] to [N, C, L]
        trend_init = trend_init.transpose(1, 2)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # N, 134, output_len
        pred_y = seasonal_output + trend_output
        return pred_y[:, :, -self.output_len:]  # Take the last output_len predictions

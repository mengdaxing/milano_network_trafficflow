import torch.nn as nn


# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, output_channels):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output_channels = output_channels

        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=(kernel_size-1) * dilation_size // 2),
                       nn.BatchNorm1d(out_channels),
                       nn.ReLU(),
                       nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)
        self.regressor = nn.Linear(num_channels[-1], output_size * output_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.regressor(out)
        out = out.reshape(-1, self.output_channels, self.output_size)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(MultiLayerPerceptron, self).__init__()
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear1 = nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        # x = F.relu(x)
        # x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# if __name__ == '__main__':
#     batch_size = 16
#     input_size = 400
#     hidden_size = 128
#     output_size = 2
#     net = MultiLayerPerceptron(input_size, hidden_size, output_size)
#     x = torch.randn(batch_size, input_size)
#     print('x.shape:', x.shape)
#     output = net(x)
#     print('output:', output.shape)

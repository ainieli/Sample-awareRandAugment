from torch.nn import Module, Dropout, Identity

from models.layers import NormAct, Conv2d


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride, dropout):
        super(BasicBlock, self).__init__()
        out_channels = channels * self.expansion

        self.norm_act1 = NormAct(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm_act2 = NormAct(out_channels)
        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = x
        x = self.norm_act1(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm_act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + shortcut



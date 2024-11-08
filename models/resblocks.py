from torch.nn import Module, Identity

from models.layers import Conv2d, Act


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride):
        super(BasicBlock, self).__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, norm='bn', act='relu')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='bn')

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, norm='bn')
        else:
            self.shortcut = Identity()

        self.act = Act()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride, dilation=1):
        super(Bottleneck, self).__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='bn', act='relu')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='bn', act='relu', dilation=dilation)
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='bn')

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, norm='bn')
        else:
            self.shortcut = Identity()
        self.act = Act()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x
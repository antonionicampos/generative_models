from torch import nn


class DeepConvolutionalCritic(nn.Module):
    def __init__(self):
        super(DeepConvolutionalCritic, self).__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.flat = nn.Flatten()
        self.linear = nn.Linear(256 * 2 * 2, 1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batchnorm1(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm2(self.conv3(x)))
        
        x = self.flat(x)
        x = self.linear(x)
        return x

from torch import nn


class Generate(nn.Module):

    def __init__(self, noise=300):
        super(Generate, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(noise, 64 * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.classifier(x)
        return out


class Discriminate(nn.Module):

    def __init__(self):
        super(Discriminate, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64 * 4, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.classifier(x)
        out = out.view(-1, 1).squeeze(1)
        return out

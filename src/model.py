import torch
from torch import nn
from torch.nn import functional as F

class Inception3D(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super().__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv3d(branch_channels, branch_channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv3d(branch_channels, branch_channels, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d((1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch_1(x), self.branch_3(x), self.branch_5(x), self.branch_pool(x)],
            dim=1,
        )

class SolarWindBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend([
                Inception3D(in_channels, 32),
                nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ])
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)
        self.image_lstm = nn.LSTM(
            input_size=128 * 4 * 4,
            hidden_size=128,
            batch_first=True,
        )
        self.wind_encoder = nn.Sequential(
            nn.Linear(20, 128), nn.SELU(inplace=True),
            nn.Linear(128, 64), nn.SELU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(inplace=True), nn.Linear(64, 12)
        )

    def forward(self, images, wind):
        image_features = images.permute(0, 2, 1, 3, 4).contiguous()
        image_features = self.stem(image_features)
        image_features = self.image_encoder(image_features)
        image_features = image_features.permute(0, 2, 1, 3, 4).flatten(2)
        
        _, (hidden, _) = self.image_lstm(image_features)
        image_features = F.relu(hidden[-1])
        wind_features = self.wind_encoder(wind)
        return self.head(torch.cat([image_features, wind_features], dim=1))
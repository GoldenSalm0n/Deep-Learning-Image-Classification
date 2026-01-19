import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # 📉 ENCODER
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            # 64 -> 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            # 32 -> 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            # 16 -> 8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            # 🔥 НОВИЙ ШАР: 8 -> 4 (Дуже сильне стиснення!)
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU()
        )

        # 📈 DECODER
        self.decoder = nn.Sequential(
            # 🔥 РОЗГОРТАЄМО: 4 -> 8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            # 64 -> 128
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
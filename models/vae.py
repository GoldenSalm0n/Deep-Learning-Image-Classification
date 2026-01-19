import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, img_channels=3, z_dim=128):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),  # -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # -> 8x8
            nn.ReLU()
        )

        # Картинку у вектор
        self.flatten_dim = 256 * 8 * 8

        # --- 2. THE SPLIT (Роздвоєння) ---
        # Замість одного шару bottleneck, у нас їх два паралельних!
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)  # Вчить середнє (центр хмари)
        self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)  # Вчить розкид (розмір хмари)

        # Шар, щоб вектор назад у картинку
        self.fc_decode = nn.Linear(z_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Вихід 0...1
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        Ми не можемо просто зробити random(), бо по ньому не проходить градієнт.
        => z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)  # Перетворюємо log_var у стандартне відхилення
        eps = torch.randn_like(std)  # Випадковий шум
        return mu + eps * std  # Зсуваємо шум на наше середнє і розтягуємо

    def forward(self, x):
        # 1. Encode
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)  # Випрямляємо

        # 2. Отримуємо параметри розподілу
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)

        # 3. Випадково обираємо точку
        z = self.reparameterize(mu, logvar)

        # 4. Decode
        z_expanded = self.fc_decode(z)
        z_reshaped = z_expanded.view(-1, 256, 8, 8)  # Повертаємо форму куба
        reconstruction = self.decoder(z_reshaped)

        # Повертаємо все (для Loss function)
        return reconstruction, mu, logvar
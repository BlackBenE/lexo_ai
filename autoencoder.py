import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=9):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_autoencoder(images, epochs=50, lr=1e-3):
    
    images = images.astype(np.float32) / 255.0
    print("images.shape:", images.shape)  # Ajout pour debug
    images_tensor = torch.tensor(images).view(-1, 64*64*3)
    
    
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(images_tensor)
        loss = criterion(outputs, images_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
    return model

def compute_reconstruction_errors(model, images):
    
    images = images.astype(np.float32) / 255.0
    images_tensor = torch.tensor(images).view(-1, 64*64*3)
    model.eval()
    with torch.no_grad():
        reconstructions = model(images_tensor)
        errors = ((reconstructions - images_tensor) ** 2).mean(dim=1)
    return errors.numpy()  # un tableau avec une erreur par image

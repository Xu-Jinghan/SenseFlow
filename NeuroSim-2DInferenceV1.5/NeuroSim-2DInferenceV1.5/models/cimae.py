import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.a = nn.Conv2d(1, 64, kernel_size=7, stride=(2,2), padding=(3,3))  # 96 -> 48, downsampling
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(3, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 48->96
        #     nn.ReLU(),
            
        #     # TransposeConv 64->32, kernel=3
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 96->96
        #     nn.ReLU(),
            
        #     # Conv 32->1, kernel=3
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # 96->96
        #     # nn.Sigmoid(),  # Output range [0, 1]
        # )

    def forward(self, x):
        x = self.a(x)
        # x = self.decoder(x)
        return x
    
def cimae(**kwargs):
    model = Autoencoder()
    return model

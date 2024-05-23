import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

LATENT_CODE_NUM = 32
class VAE_decoder(nn.Module):
    def __init__(self, infeat_size):
        super(VAE_decoder, self).__init__()
        self.infeat_size = infeat_size
        self.fc11 = nn.Linear(self.infeat_size[0] * self.infeat_size[1] * self.infeat_size[2], LATENT_CODE_NUM)
        self.fc12 = nn.Linear(self.infeat_size[0] * self.infeat_size[1] * self.infeat_size[2], LATENT_CODE_NUM)
        self.fc2 = nn.Linear(LATENT_CODE_NUM, self.infeat_size[0] * self.infeat_size[1] * self.infeat_size[2])
        
        self.decoder = nn.Sequential(                
                nn.ConvTranspose2d(self.infeat_size[0], 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
                )

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + eps * torch.exp(logvar/2)
        
        return z
    
    def forward(self, f):
        # f.shape = batch_s, 8, 7, 7
        f1, f2 = f, f
        mu = self.fc11(f1.view(f1.size(0),-1))     # batch_s, latent
        logvar = self.fc12(f2.view(f2.size(0),-1)) # batch_s, latent
        z = self.reparameterize(mu, logvar)      # batch_s, latent      
        out3 = self.fc2(z).view(z.size(0), self.infeat_size[0], self.infeat_size[1], self.infeat_size[2])    # batch_s, 8, 7, 7
        
        return self.decoder(out3), mu, logvar


def vae_loss_func(recon_x, x, mu, logvar):
    x = x.resize_(recon_x.size(0),recon_x.size(1),recon_x.size(2),recon_x.size(3))
    #bce_loss = nn.BCELoss()
    #bce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.MSELoss()
    BCE = bce_loss(recon_x, x)
    #BCE = bce_loss(torch.sigmoid(recon_x), torch.sigmoid(x))
    #BCE = F.binary_cross_entropy(recon_x, x)
    #BCE = 0
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = 0
    return BCE+KLD


from torch import nn
import torch.nn.functional as F
import torch


# -----------------------------------------------------------------
#  Variational auto-encoder
# -----------------------------------------------------------------
class VAE(nn.Module):

    def __init__(self, features_in=1500):

        super(VAE, self).__init__()

        # encoder components
        self.fc1 = nn.Linear(features_in, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc31 = nn.Linear(300, 10)
        self.fc32 = nn.Linear(300, 10)

        # decoder components
        self.fc4 = nn.Linear(10, 300)
        self.fc5 = nn.Linear(300, 1000)
        self.fc6 = nn.Linear(1000, features_in)

    def encode(self, x):

        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, log_variance):

        std = torch.exp(0.5*log_variance)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        h4 = F.leaky_relu(self.fc4(z))
        h5 = F.leaky_relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):

        mu, log_variance = self.encode(x.view(-1, 1500))
        z = self.reparameterize(mu, log_variance)
        return self.decode(z), mu, log_variance


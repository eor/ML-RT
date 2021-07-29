from torch import nn
import torch.nn.functional as F
import torch


# -----------------------------------------------------------------
#  Variational auto-encoder
# -----------------------------------------------------------------
class VAE1(nn.Module):

    def __init__(self, features_in=1500):

        super(VAE1, self).__init__()

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

    def reparametrise(self, mu, log_variance):

        std = torch.exp(0.5*log_variance)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        h4 = F.leaky_relu(self.fc4(z))
        h5 = F.leaky_relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):

        mu, log_variance = self.encode(x.view(-1, 1500))
        z = self.reparametrise(mu, log_variance)
        return self.decode(z), mu, log_variance


# -----------------------------------------------------------------
#  Conditional variational auto-encoder
# -----------------------------------------------------------------
class CVAE1(nn.Module):

    # -------------------------------------------------------------
    #  idea here:
    #  same  as VAE but input and latent vector are concatenated,
    #  i.e. conditioned, with the parameter vector
    # -------------------------------------------------------------

    def __init__(self, features_in=1500, nParams=5):

        super(CVAE1, self).__init__()

        nLatent = 20

        # encoder
        ## 1st hidden layer
        self.fc1 = nn.Linear(features_in+nParams, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)

        ## 2nd hidden layer
        self.fc2 = nn.Linear(1000, 300)
        self.fc2_bn = nn.BatchNorm1d(300)

        self.fc31 = nn.Linear(300, nLatent)
        self.fc32 = nn.Linear(300, nLatent)

        # decoder
        self.fc4 = nn.Linear(nLatent+nParams, 300)

        self.fc5 = nn.Linear(300, 1000)
        self.fc5_bn = nn.BatchNorm1d(1000)

        self.fc6 = nn.Linear(1000, features_in)

    # ---------------------------------------------------------
    # Methods
    # ---------------------------------------------------------
    def encode(self, x):

        out = self.fc1(x)
        out = self.fc1_bn(out)
        out = F.leaky_relu(out)

        out = self.fc2(out)
        out = self.fc2_bn(out)
        out = F.leaky_relu(out)

        return self.fc31(out), self.fc32(out)

    def reparameterize(self, mu, log_variance):

        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)

        return mu + eps*std

    def decode(self, z):

        out = self.fc4(z)
        out = F.leaky_relu(out)


        out = self.fc5(out)
        out = self.fc5_bn(out)
        out = F.leaky_relu(out)

        return self.fc6(out)

    def forward(self, x, y):

        prof = x.view(-1, 1500)         # TODO: At some point the 1500 & 5 should no longer be hard-coded
        para = y.view(-1, 5)

        pp = torch.cat((prof, para), 1)  # dimension 1 is profile length plus number of parameters,
                                         # dimension 0 is the batch (size)

        mu, log_variance = self.encode(pp)    # 1505

        z = self.reparameterize(mu, log_variance)

        zp = torch.cat((z, para), 1)      # condition latent vector with parameters

        return self.decode(zp), mu, log_variance #, self.trans_predict(z)
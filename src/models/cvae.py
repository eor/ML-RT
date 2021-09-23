from torch import nn
import torch.nn.functional as F
import torch


# -----------------------------------------------------------------
#  Variational auto-encoder
# -----------------------------------------------------------------
class VAE1(nn.Module):

    # TODO: test this method

    def __init__(self, conf):
        super(VAE1, self).__init__()

        # encoder components
        self.fc1 = nn.Linear(conf.profile_len, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc31 = nn.Linear(300, 10)
        self.fc32 = nn.Linear(300, 10)

        # decoder components
        self.fc4 = nn.Linear(10, 300)
        self.fc5 = nn.Linear(300, 1000)
        self.fc6 = nn.Linear(1000, conf.profile_len)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparametrise(self, mu, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.leaky_relu(self.fc4(z))
        h5 = F.leaky_relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, y=None):
        # mu, log_variance = self.encode(x.view(-1, 1500))
        mu, log_variance = self.encode(x)
        z = self.reparametrise(mu, log_variance)
        return self.decode(z), mu, log_variance


# -----------------------------------------------------------------
#  Conditional variational auto-encoder
# -----------------------------------------------------------------
class CVAE1(nn.Module):
    """
    Idea here:
    Same models as VAE but input profile and latent vector are concatenated, i.e. conditioned,
    with the parameter vector.

    """

    use_batch_norm = False
    use_dropout = False

    # TODO: 1) refactor 2) add drop out layers

    def __init__(self, conf):
        super(CVAE1, self).__init__()

        n_latent_space = conf.latent_dim

        features_in = conf.profile_len
        n_parameters = conf.n_parameters
        self.use_batch_norm = conf.batch_norm
        self.use_dropout = conf.dropout

        # encoder
        ## 1st hidden layer
        self.fc1 = nn.Linear(features_in + n_parameters, 1000)
        if self.use_batch_norm:
            self.fc1_bn = nn.BatchNorm1d(1000)

        ## 2nd hidden layer
        self.fc2 = nn.Linear(1000, 300)
        if self.use_batch_norm:
            self.fc2_bn = nn.BatchNorm1d(300)

        self.fc31 = nn.Linear(300, n_latent_space)
        self.fc32 = nn.Linear(300, n_latent_space)

        # decoder
        self.fc4 = nn.Linear(n_latent_space + n_parameters, 300)

        self.fc5 = nn.Linear(300, 1000)
        if self.use_batch_norm:
            self.fc5_bn = nn.BatchNorm1d(1000)

        self.fc6 = nn.Linear(1000, features_in)

    # ---------------------------------------------------------
    # Methods
    # ---------------------------------------------------------
    def encode(self, x):

        out = self.fc1(x)

        if self.use_batch_norm:
            out = self.fc1_bn(out)

        out = F.leaky_relu(out)

        out = self.fc2(out)

        if self.use_batch_norm:
            out = self.fc2_bn(out)

        out = F.leaky_relu(out)

        return self.fc31(out), self.fc32(out)

    def reparameterize(self, mu, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        out = self.fc4(z)
        out = F.leaky_relu(out)

        out = self.fc5(out)

        if self.use_batch_norm:
            out = self.fc5_bn(out)

        out = F.leaky_relu(out)

        return self.fc6(out)

    def forward(self, x, y):

        profiles = x
        parameters = y

        # concatenate input data:
        # profiles + parameter vectors (along dimension 1, the profile length,
        # dimension 0 is the batch size)
        cond_profiles = torch.cat((profiles, parameters), 1)

        # pass conditioned input through encoder
        mu, log_variance = self.encode(cond_profiles)

        # re-parametrisation to obtain latent vector z
        z = self.reparameterize(mu, log_variance)

        # condition latent vector with the parameters
        cond_z = torch.cat((z, parameters), 1)

        # run conditioned latent vector through decoder and return
        return self.decode(cond_z), mu, log_variance



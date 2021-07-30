from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm):
            layers = [nn.Linear(features_in, features_out)]
            if normalise:
                layers.append(nn.BatchNorm1d(features_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.latent_dim + conf.n_parameters, 128, normalise=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(conf.profile_len)),
            # nn.Tanh()
        )

    def forward(self, noise, parameters):

        # condition latent vector with parameters, i.e. concatenate latent
        # vector with parameter vector to produce generator input
        generator_input = torch.cat((noise, parameters), -1)
        generator_output = self.model(generator_input)

        return generator_output


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()

        def block(features_in, features_out, dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]
            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.profile_len + conf.n_parameters, 1024, dropout=False),
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, profile, parameters):

        # concatenate profile and parameter vector to produce conditioned input
        # Inputs: profile.shape -  [batch_size, profile_len]
        #         parameters.shape - [batch_size, n_parameters]

        discriminator_input = torch.cat((profile, parameters), 1)

        validity = self.model(discriminator_input)

        return validity
    

from torch import nn
import torch


# -----------------------------------------------------------------
# Generator 1: 4 hidden layers, features BN (Based on MLP (without dropout))
# -----------------------------------------------------------------
class Generator1(nn.Module):
    def __init__(self, conf):
        super(Generator1, self).__init__()

        def block(features_in, features_out, use_batch_norm=conf.batch_norm):
            layers = [nn.Linear(features_in, features_out)]

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(features_out))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.latent_dim + conf.n_parameters, 128, use_batch_norm=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(conf.profile_len))
            # nn.Tanh()  <--- NOPE!!!!!!!!!
        )

    def forward(self, noise, parameters):

        # condition latent vector with parameters, i.e. concatenate latent
        # vector with parameter vector to produce generator input
        generator_input = torch.cat((noise, parameters), -1)
        generator_output = self.model(generator_input)

        return generator_output


# -----------------------------------------------------------------
# Generator 2: simpler model, 2 hidden layers, features BN
# -----------------------------------------------------------------
class Generator2(nn.Module):
    def __init__(self, conf):
        super(Generator2, self).__init__()

        def block(features_in, features_out, use_batch_norm=conf.batch_norm):
            layers = [nn.Linear(features_in, features_out)]

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(features_out))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.latent_dim + conf.n_parameters, 256, use_batch_norm=False),
            *block(256, 3000),
            nn.Linear(3000, int(conf.profile_len))
        )

    def forward(self, noise, parameters):

        # condition latent vector with parameters, i.e. concatenate latent
        # vector with parameter vector to produce generator input
        generator_input = torch.cat((noise, parameters), -1)
        generator_output = self.model(generator_input)

        return generator_output


# -----------------------------------------------------------------
# Discriminator 1: 5 hidden layers, features dropout
# -----------------------------------------------------------------
class Discriminator1(nn.Module):
    def __init__(self, conf):
        super(Discriminator1, self).__init__()

        def block(features_in, features_out, use_dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]

            if use_dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.profile_len + conf.n_parameters, 1024, use_dropout=False),
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            *block(64, 1, use_dropout=False),
        )

    def forward(self, profiles, parameters):
        # concatenate profile and parameter vector to produce conditioned input
        # Inputs: profile.shape -  [batch_size, profile_len]
        #         parameters.shape - [batch_size, n_parameters]

        # user hasn't passed the already concatenated input, so concat it
        if parameters is not None:
            discriminator_input = torch.cat((profiles, parameters), 1)
            validity = self.model(discriminator_input)
        else:
            validity = self.model(profiles)

        return validity


# -----------------------------------------------------------------
# Discriminator 2: simpler model, 2 hidden layer, features dropout
# -----------------------------------------------------------------
class Discriminator2(nn.Module):
    def __init__(self, conf):
        super(Discriminator2, self).__init__()

        def block(features_in, features_out, use_dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]

            if use_dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.profile_len + conf.n_parameters, 512, use_dropout=False),
            *block(512, 256),
            *block(256, 128),
            *block(128, 1, use_dropout=False)
        )

    def forward(self, profiles, parameters):

        # concatenate profile and parameter vector to produce conditioned input
        # Inputs: profile.shape -  [batch_size, profile_len]
        #         parameters.shape - [batch_size, n_parameters]

        if parameters is not None:
            discriminator_input = torch.cat((profiles, parameters), 1)
            validity = self.model(discriminator_input)
        else:
            validity = self.model(profiles)

        return validity

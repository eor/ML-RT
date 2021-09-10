from torch import nn
import torch


# -----------------------------------------------------------------
# Generator 1: 4 hidden layers, features BN
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
# Generator 3: based on LSTM 1
# -----------------------------------------------------------------
class Generator3(nn.Module):
    def __init__(self, conf, device):
        super(Generator3, self).__init__()

        self.input_size = conf.n_parameters + conf.latent_dim
        self.seq_len = conf.profile_len
        self.device = device
        self.num_layers = 1

        def block(features_in, features_out):
            layers = [nn.Linear(features_in, features_out)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Args: expects input of shape: (batch_size, n_parameters)
        #       output of shape: (batch_size, 2*time_series_length)
        self.linear_model = nn.Sequential(
            *block(self.input_size, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, self.seq_len),
            nn.Linear(self.seq_len, self.seq_len * 2)
        )

        # Args: expects input of shape: (batch_size, time_series_length, input_size)
        #       output of shape: (batch_size, time_series_length, 2*hidden_size): x2 because, bidirectional_lstm
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True, bidirectional=True, num_layers=self.num_layers)

        # Args: expects input of shape: (batch_size, 2*2*time_series_length)
        #       output of shape: (batch_size, time_series_length)
        self.out_layer = nn.Linear(2 * 2 * self.seq_len, self.seq_len)

    def forward(self, noise, parameters):

        # initialise hidden states for the lstm
        generator_input = torch.cat((noise, parameters), -1)

        (hidden_state, cell_state) = self.init_hidden_state(batch_size=generator_input.size()[0])
        x = self.linear_model(generator_input)
        # x.size(): (batch_size, 2*time_series_length) => (batch_size, 2*time_series_length, input_size)
        x = torch.unsqueeze(x, dim=2)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        # x.size(): (batch_size, 2*time_series_length, 2*input_size) => (batch_size, 2*2*time_series_length)
        x = x.reshape(x.size()[0], -1)
        x = self.out_layer(x)
        return x

    def init_hidden_state(self, batch_size):
        # x2 because, bidirectional lstm
        hidden_state = torch.zeros(2 * self.num_layers, batch_size, 1, device=self.device)
        cell_state = torch.zeros(2 * self.num_layers, batch_size, 1, device=self.device)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return (hidden_state, cell_state)


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
            nn.Linear(64, 1)
        )

    def forward(self, profiles, parameters):

        # concatenate profile and parameter vector to produce conditioned input
        # Inputs: profile.shape -  [batch_size, profile_len]
        #         parameters.shape - [batch_size, n_parameters]

        discriminator_input = torch.cat((profiles, parameters), 1)

        validity = self.model(discriminator_input)

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
            *block(conf.profile_len + conf.n_parameters, 1024, use_dropout=False),
            *block(1024, 128),
            nn.Linear(128, 1),
        )

    def forward(self, profiles, parameters):

        # concatenate profile and parameter vector to produce conditioned input
        # Inputs: profile.shape -  [batch_size, profile_len]
        #         parameters.shape - [batch_size, n_parameters]

        discriminator_input = torch.cat((profiles, parameters), 1)

        validity = self.model(discriminator_input)

        return validity
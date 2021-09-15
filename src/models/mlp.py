from torch import nn
import torch


class MLP1(nn.Module):
    def __init__(self, conf):
        super(MLP1, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):

            layers = [nn.Linear(features_in, features_out)]

            # Different order of BN, Dropout, and non-linearity should be explored!
            # From the literature it seems like there is no canonical way of doing it.

            if normalise:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(conf.n_parameters, 64, normalise=False, dropout=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(conf.profile_len))
        )

    def forward(self, parameters):

        return self.model(parameters)


class MLP2(nn.Module):
    def __init__(self, conf):
        super(MLP2, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):

            layers = [nn.Linear(features_in, features_out)]

            if normalise:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(conf.n_parameters, 256, normalise=False, dropout=False),
            *block(256, 3000),
            nn.Linear(3000, int(conf.profile_len))
        )

    def forward(self, parameters):

        return self.model(parameters)


class MLP3(nn.Module):
    def __init__(self, conf):
        super(MLP3, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):

            layers = [nn.Linear(features_in, features_out)]

            # Different order of BN, Dropout, and non-linearity should be explored!
            # From the literature it seems like there is no canonical way of doing it.

            if normalise:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(conf.n_parameters, 32, normalise=False, dropout=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            *block(2048, 4096),
            nn.Linear(4096, int(conf.profile_len))
        )

    def forward(self, parameters):

        return self.model(parameters)
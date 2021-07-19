from torch import nn
#import torch.nn.functional as F
import torch


class MLP1(nn.Module):
    def __init__(self, conf):
        super(MLP, self).__init__()

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
            nn.Linear(1024, int(conf.profile_len)),

        )

    def forward(self, parameters):

        return self.model(parameters)


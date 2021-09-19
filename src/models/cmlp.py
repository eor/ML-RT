from torch import nn
import torch

class CMLP(nn.Module):
    def __init__(self, conf, device):
        super(CMLP, self).__init__()

        def block(features_in, features_out, normalise=False, dropout=False):

            layers = [nn.Linear(features_in, features_out)]
            
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
            *block(1024, int(conf.profile_len)),
#             nn.Linear(1024, int(conf.profile_len))
        )
        
        self.out_layer_H_II = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_T = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_He_II = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_He_III = nn.Linear(int(conf.profile_len), int(conf.profile_len))

    def forward(self, parameters):
        x = self.model(parameters)
        x_H_II = self.out_layer_H_II(x)
        x_T = self.out_layer_T(x)
        x_He_II = self.out_layer_He_II(x)
        x_He_III = self.out_layer_He_III(x)

        return x_H_II, x_T, x_He_II, x_He_III


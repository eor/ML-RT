from torch import nn
import torch

class CMLP(nn.Module):
    def __init__(self, conf, device):
        super(CMLP, self).__init__()
        self.conf = conf
        def block(features_in, features_out):
            layers = [nn.Linear(features_in, features_out)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(conf.n_parameters, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, int(4 * conf.profile_len))
        )

        self.out_layer_H_II = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_T = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_He_II = nn.Linear(int(conf.profile_len), int(conf.profile_len))
        self.out_layer_He_III = nn.Linear(int(conf.profile_len), int(conf.profile_len))

    def forward(self, parameters):
        x = self.model(parameters)

        x_H_II = x[:, 0: int(self.conf.profile_len)]
        x_T = x[:, int(self.conf.profile_len): 2 * int(self.conf.profile_len)]
        x_He_II = x[:, 2 * int(self.conf.profile_len): 3 * int(self.conf.profile_len)]
        x_He_III = x[:, 3 * int(self.conf.profile_len): 4 * int(self.conf.profile_len)]

        x_H_II = self.out_layer_H_II(x_H_II)
        x_T = self.out_layer_T(x_T)
        x_He_II = self.out_layer_He_II(x_He_II)
        x_He_III = self.out_layer_He_III(x_He_III)

        return x_H_II, x_T, x_He_II, x_He_III

from torch import nn
import torch


class CLSTM(nn.Module):
    def __init__(self, conf, device):
        super(CLSTM, self).__init__()

        self.conf = conf
        self.device = device

        def block(features_in, features_out):

            layers = [nn.Linear(features_in, features_out)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        # Args: expects input of shape: (batch_size, n_parameters)
        #       output of shape: (batch_size, 2*profile_length)
        self.linear_model = nn.Sequential(
            *block(self.conf.n_parameters, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, self.conf.profile_len),
            nn.Linear(self.conf.profile_len, self.conf.profile_len * 2)
        )

        # Args: expects input of shape: (batch_size, profile_length, n_parameters)
        #       output of shape: (batch_size, profile_length, 2*hidden_size): x2 because, bidirectional_lstm
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=4, batch_first=True, bidirectional=True)

        # Args: expects input of shape: (batch_size, 2*2*profile_length)
        #       output of shape: (batch_size, profile_length)
        self.out_layer_H_II = nn.Linear(4 * self.conf.profile_len, self.conf.profile_len)
        self.out_layer_T = nn.Linear(4 * self.conf.profile_len, self.conf.profile_len)
        self.out_layer_He_II = nn.Linear(4 * self.conf.profile_len, self.conf.profile_len)
        self.out_layer_He_III = nn.Linear(4 * self.conf.profile_len, self.conf.profile_len)

    def forward(self, x):

        # Initialise hidden states for the LSTM
        (hidden_state, cell_state) = self.init_hidden_state(batch_size=x.size()[0], hidden_size=64, bidirectional=True)
        (hidden_state_, cell_state_) = self.init_hidden_state(batch_size=x.size()[0], hidden_size=4, bidirectional=True)

        x = self.linear_model(x)

        x = torch.unsqueeze(x, dim=2)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        x, (hidden_state_, cell_state_) = self.lstm_1(x, (hidden_state_, cell_state_))

        x_H_II = torch.stack((x[:, :, 0], x[:, :, 4]), dim=2).reshape(x.size()[0], -1)
        x_T = torch.stack((x[:, :, 1], x[:, :, 5]), dim=2).reshape(x.size()[0], -1)
        x_He_II = torch.stack((x[:, :, 2], x[:, :, 6]), dim=2).reshape(x.size()[0], -1)
        x_He_III = torch.stack((x[:, :, 3], x[:, :, 7]), dim=2).reshape(x.size()[0], -1)

        x_H_II = self.out_layer_H_II(x_H_II)
        x_T = self.out_layer_T(x_T)
        x_He_II = self.out_layer_He_II(x_He_II)
        x_He_III = self.out_layer_He_III(x_He_III)

        return x_H_II, x_T, x_He_II, x_He_III

    def init_hidden_state(self, batch_size, hidden_size, bidirectional=False):

        if bidirectional:
            hidden_state = torch.zeros(2, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(2, batch_size, hidden_size, device=self.device)
        else:
            hidden_state = torch.zeros(1, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(1, batch_size, hidden_size, device=self.device)

        # Initialisation of weights
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return (hidden_state, cell_state)

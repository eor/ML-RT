from torch import nn
import torch


class LSTM1(nn.Module):
    def __init__(self, conf, device):
        super(LSTM1, self).__init__()

        self.conf = conf

        self.device = device
        self.num_layers = 1

        def block(features_in, features_out):

            layers = [nn.Linear(features_in, features_out)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        # linear model expects input of shape: (batch_size, n_parameters)
        # output of shape: (batch_size, 2*profile_length)
        self.linear_model = nn.Sequential(
            *block(self.conf.n_parameters, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, self.conf.profile_len),
            nn.Linear(self.conf.profile_len, self.conf.profile_len * 2)
        )

        #  expects input of shape: (batch_size, profile_length, n_parameters)
        #  output is of shape: (batch_size, profile_length, 2*hidden_size): x2 because, bidirectional_lstm
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True, bidirectional=True, num_layers=self.num_layers)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=1, batch_first=True, bidirectional=True, num_layers=self.num_layers)

        # out layer expects input of shape: (batch_size, 2*2*profile_length)
        # output is of shape: (batch_size, profile_length)
        self.out_layer = nn.Linear(2 * 2 * self.conf.profile_len, self.conf.profile_len)

    def forward(self, x):

        # Initialise hidden states for the LSTM
        (hidden_state_1, cell_state_1) = self.init_hidden_state(bidirectional=True, batch_size=x.size()[0], hidden_size=64)
        (hidden_state_2, cell_state_2) = self.init_hidden_state(bidirectional=True, batch_size=x.size()[0], hidden_size=1)

        x = self.linear_model(x)

        # x.size(): (batch_size, 2*profile_length) => (batch_size, 2*profile_length, input_size)

        x = torch.unsqueeze(x, dim=2)
        x, (hidden_state_1, cell_state_1) = self.lstm_1(x, (hidden_state_1, cell_state_1))
        x, (hidden_state_2, cell_state_2) = self.lstm_2(x, (hidden_state_2, cell_state_2))

        # x.size(): (batch_size, 2*profile_length, 2*input_size) => (batch_size, 2*2*profile_length)

        x = x.reshape(x.size()[0], -1)
        x = self.out_layer(x)

        return x

    def init_hidden_state(self, bidirectional, batch_size, hidden_size):

        if bidirectional:
            # x 2 because bidirectional lstm
            hidden_state = torch.zeros(2 * self.num_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(2 * self.num_layers, batch_size, hidden_size, device=self.device)
        else:
            # x 2 because bidirectional lstm
            hidden_state = torch.zeros(self.num_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(self.num_layers, batch_size, hidden_size, device=self.device)

        # Initialisation of weights
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return (hidden_state, cell_state)

from torch import nn
import torch


class LSTM1(nn.Module):
    def __init__(self, conf, device):
        super(LSTM1, self).__init__()

        self.input_size = conf.n_parameters
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
        self.out_layer = nn.Linear(2 * 2* self.seq_len, self.seq_len)

    def forward(self, x):

        # initialise hidden states for the lstm
        (hidden_state, cell_state) = self.init_hidden_state(batch_size=x.size()[0])

        x = self.linear_model(x)
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


class LSTM2(nn.Module):
    def __init__(self, conf, device):
        super(LSTM2, self).__init__()
        self.input_size = conf.n_parameters
        # If True, becomes a bidirectional LSTM
        self.bidirectional = True
        # number of values we want to predict
        self.seq_len = conf.profile_len
        self.device = device

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
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=1, batch_first=True, bidirectional=True)

        # Args: expects input of shape: (batch_size, time_series_length, input_size)
        #       output of shape: (batch_size, time_series_length, hidden_size): x2 because, bidirectional_lstm
        self.lstm_2 = nn.LSTM(input_size=2, hidden_size=1, batch_first=True, bidirectional=False)

        # Args: expects input of shape: (batch_size, time_series_length)
        #       output of shape: (batch_size, time_series_length)
        self.out_layer_2 = nn.Linear(2*self.seq_len, self.seq_len)

    def forward(self, x):

        # initialise hidden states for the the lstm's
        (hidden_state_1, cell_state_1) = self.init_hidden_state(batch_size=x.size()[0], bidirectional=True)
        (hidden_state_2, cell_state_2) = self.init_hidden_state(batch_size=x.size()[0], bidirectional=False)

        x = self.linear_model(x)
        # x.size(): (batch_size, 2*time_series_length) => (batch_size, 2*time_series_length, input_size)
        x = torch.unsqueeze(x, dim=2)
        x, (hidden_state_1, cell_state_1) = self.lstm_1(x, (hidden_state_1, cell_state_1))
        x, (hidden_state_2, cell_state_2) = self.lstm_2(x, (hidden_state_2, cell_state_2))
        # x.size(): (batch_size, time_series_length, input_size) => (batch_size, time_series_length)
        x = x.reshape(x.size()[0], -1)
        x = self.out_layer_2(x)

        return x

    def init_hidden_state(self, batch_size, bidirectional=False):
        if bidirectional:
            hidden_state = torch.zeros(2, batch_size, 1, device=self.device)
            cell_state = torch.zeros(2, batch_size, 1, device=self.device)
        else:
            hidden_state = torch.zeros(1, batch_size, 1, device=self.device)
            cell_state = torch.zeros(1, batch_size, 1, device=self.device)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return (hidden_state, cell_state)

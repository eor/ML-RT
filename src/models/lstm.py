from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, conf, device):
        super(LSTM, self).__init__()
        self.input_size = conf.n_parameters
        # If False, then the layer does not use bias weights b_ih and b_hh
        self.bias = True
        # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        self.batch_first = True
        # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.dropout = conf.dropout_value
        # If True, becomes a bidirectional LSTM
        self.bidirectional = True
        # number of values we want to predict
        self.seq_len = conf.profile_len
        # batch_size
        # self.batch_size = conf.batch_size
        self.batch_size = conf.batch_size
        self.device = device

        # layer_params
        # self.layer_params = [self.lstm_input, 128, 64, 16, self.lstm_out]
        # self.layer_params = [self.lstm_input, self.lstm_out]
        # Number of hidden layers
        # self.num_layers = len(self.layer_params) - 1
        self.num_layers = 2
        self.linear_model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.seq_len),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.seq_len, self.seq_len * 2),
        )
        # lstm_input
        self.lstm_input = 1
        # Hidden dimensions ?????
        self.lstm_out = 1

        self.lstm = nn.LSTM(
            input_size = self.lstm_input,
            hidden_size = self.lstm_out,
            bias = self.bias,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional,
            num_layers = self.num_layers
        )
        
        if self.bidirectional:
            self.out_layer = nn.Linear(2 * 2 * self.seq_len, self.seq_len)
        else:
            self.out_layer = nn.Linear(2 * self.seq_len, self.seq_len)


    def forward(self, x):
        self.batch_size = x.size()[0]
        (hidden_state, cell_state) = self.init_hidden_state(n_layers = self.num_layers, batch_size = self.batch_size, hidden_size = self.lstm_out)
        x = self.linear_model(x)
        x = torch.unsqueeze(x,dim=2)
        x, (hidden_state,cell_state) = self.lstm(x, (hidden_state,cell_state))
        x = torch.squeeze(x,dim=2)
        if self.bidirectional:
            x = x.reshape(x.size()[0],-1)
        x = self.out_layer(x)
        return x


    def init_hidden_state(self, n_layers, batch_size, hidden_size):
        if self.bidirectional:
            hidden_state = torch.zeros(2 * n_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(2 * n_layers, batch_size, hidden_size, device=self.device)
        else:
            hidden_state = torch.zeros(n_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(n_layers, batch_size, hidden_size, device=self.device)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return [hidden_state,cell_state]
    
    
class LSTM2(nn.Module):
    def __init__(self, conf, device):
        super(LSTM2, self).__init__()
        self.input_size = conf.n_parameters        
        # If False, then the layer does not use bias weights b_ih and b_hh
        self.bias = True
        # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        self.batch_first = True
        # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.dropout = conf.dropout_value
        # If True, becomes a bidirectional LSTM
        self.bidirectional = True
        # number of values we want to predict
        self.seq_len = conf.profile_len
        # batch_size
        # self.batch_size = conf.batch_size
        self.batch_size = conf.batch_size
        self.device = device
        
        self.num_layers = 1
        self.linear_model = nn.Sequential(
            nn.Linear(self.input_size, 64),
#             nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.seq_len),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.seq_len, self.seq_len*2)
        )
        # lstm_input
        self.lstm_input = 1
        # Hidden dimensions ?????
        self.lstm_out = 1
        
        self.lstm = nn.LSTM(
            input_size = self.lstm_input,
            hidden_size = self.lstm_out,
            bias = self.bias,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional,
            num_layers = self.num_layers
        )
        
        if self.bidirectional:
            self.out_layer = nn.Linear(2 * 2 * self.seq_len, self.seq_len)
        else:
            self.out_layer = nn.Linear(2 * self.seq_len, self.seq_len)
        
        self.lstm_1 = nn.LSTM(
            input_size = self.lstm_input,
            hidden_size = self.lstm_out,
            bias = self.bias,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional,
            num_layers = self.num_layers
        )
        
        if self.bidirectional:
            self.out_layer_1 = nn.Linear(2 * self.seq_len, self.seq_len)
        else:
            self.out_layer_1 = nn.Linear(self.seq_len, self.seq_len)
        

    def forward(self, x):
        self.batch_size = x.size()[0]
        (hidden_state, cell_state) = self.init_hidden_state(n_layers = self.num_layers, batch_size = self.batch_size, hidden_size = self.lstm_out)
        (hidden_state_1, cell_state_1) = self.init_hidden_state(n_layers = self.num_layers, batch_size = self.batch_size, hidden_size = self.lstm_out)
        
        x = self.linear_model(x)
        x = torch.unsqueeze(x,dim=2)
        x, (hidden_state,cell_state) = self.lstm(x, (hidden_state,cell_state))
        x = torch.squeeze(x,dim=2)
        if self.bidirectional:
            x = x.reshape(x.size()[0],-1)
        x = self.out_layer(x)
        x = torch.unsqueeze(x,dim=2)
        x, (hidden_state_1,cell_state_1) = self.lstm_1(x, (hidden_state_1,cell_state_1))
        x = torch.squeeze(x,dim=2)
        if self.bidirectional:
            x = x.reshape(x.size()[0],-1)
        x = self.out_layer_1(x)
        
        return x
        

    def init_hidden_state(self, n_layers, batch_size, hidden_size):
        if self.bidirectional:
            hidden_state = torch.zeros(2 * n_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(2 * n_layers, batch_size, hidden_size, device=self.device)
        else:
            hidden_state = torch.zeros(n_layers, batch_size, hidden_size, device=self.device)
            cell_state = torch.zeros(n_layers, batch_size, hidden_size, device=self.device)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return [hidden_state,cell_state]
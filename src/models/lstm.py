from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, conf):
        super(LSTM, self).__init__()
        self.input_size = conf.n_parameters        
        # If False, then the layer does not use bias weights b_ih and b_hh
        self.bias = True
        # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        self.batch_first = True
        # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.dropout = conf.dropout_value
        # If True, becomes a bidirectional LSTM
        self.bidirectional = 1
        # number of values we want to predict
        self.seq_len = conf.profile_len
        # batch_size
        # self.batch_size = conf.batch_size
        self.batch_size = conf.batch_size
        
        # lstm_input
        self.lstm_input = 1
        # Hidden dimensions ?????
        self.lstm_out = 1
        # layer_params
        self.layer_params = [self.lstm_input, 128, 64, 16, self.lstm_out]
        # Number of hidden layers
        self.num_layers = len(self.layer_params) - 1
        
        self.linear_model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, self.lstm_input)
        )

        self.lstm_layers = []
        for i in range(self.num_layers):
            self.lstm_cell = nn.LSTMCell(
                    input_size = self.layer_params[i],
                    hidden_size = self.layer_params[i+1],
                    bias = self.bias            
                )
            self.lstm_layers.append(self.lstm_cell)
            
        self.out_layer = nn.Linear(self.seq_len, self.seq_len)
        

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.init_hidden_state(self.batch_size, hidden_size = self.layer_params[i+1]))

        # cell_states = [cell_state]

        # pass the parameters to sequential model to get first output
        x = self.linear_model(x)
        # print('shape of linear_model output:',x.shape)
        out_lstm = torch.empty((self.batch_size, 0))
        # print(out_lstm.size())
        for i in range(self.seq_len):
            for j in range(self.num_layers):
                hidden_states[j][0], hidden_states[j][1] = self.lstm_layers[j](x, (hidden_states[j][0], hidden_states[j][1]))
                x = hidden_states[j][0]
            out_lstm = torch.cat((out_lstm, x),dim=1)

        x = self.out_layer(out_lstm)
        return x
        

    def init_hidden_state(self, batch_size, hidden_size):
        hidden_state = torch.zeros(batch_size, hidden_size)
        cell_state = torch.zeros(batch_size, hidden_size)

        # Weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        return [hidden_state,cell_state]
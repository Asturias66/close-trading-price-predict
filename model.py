import torch
import torch.nn as nn
import torch.nn.init as init

def global_average_pooling(x):
    return x.mean(dim=(-1))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1d1 = nn.Conv1d(200, 64, 3,padding=1)
        self.conv1d2 = nn.Conv1d(64, 64, 3,padding=1)
        self.conv1d3 = nn.Conv1d(64, 128, 3,padding=1)
        self.conv1d4 = nn.Conv1d(128, 128, 3,padding=1)
        self.conv1d5 = nn.Conv1d(128, 256, 3,padding=1)
        self.conv1d6 = nn.Conv1d(256, 256, 3,padding=1)
        self.conv1d7 = nn.Conv1d(256, 256, 3,padding=1)

        self.pool1d1 = nn.AvgPool1d(2)
        self.pool1d2 = nn.AvgPool1d(2)
        self.pool1d3 = nn.AvgPool1d(2)

        self.linear1 = nn.Linear(256, 32)
        self.linear2 = nn.Linear(32, 200)

    def forward(self, inputs):
        x = self.conv1d1(inputs)
        x = self.conv1d2(x)
        x = self.pool1d1(x)

        x = self.conv1d3(x)
        x = self.conv1d4(x)
        x = self.pool1d2(x)

        x = self.conv1d5(x)
        x = self.conv1d6(x)
        x = self.conv1d7(x)

        out = global_average_pooling(x)

        out = self.linear1(out)
        out = nn.ReLU()(out)

        output = self.linear2(out)

        return output

class CNNLSTMModel(nn.Module):
    def __init__(self, window=60, dim=200, lstm_units=57, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 3,padding=1)
        self.act1 = nn.Sigmoid()
        self.pool1d2 = nn.AvgPool1d(2)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units, 200)
        self.act4 = nn.Sigmoid()
    def forward(self, x):
        # x = x.transpose(-1, -2)
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.pool1d2(x)  # bs, lstm_units, 1
        x = self.drop(x)
        # x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        # x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = global_average_pooling(x)
        x = self.cls(x)
        x = self.act4(x)
        return x

        # This function takes an input and predicts the class, (0 or 1)
    def predict(self, x):
        # Apply softmax to output
        pred = nn.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

class CNNLSTMModel_ECA(nn.Module):
    def __init__(self, window=60, dim=200, lstm_units=57, num_layers=2):
        super(CNNLSTMModel_ECA, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 3,padding=1)
        self.act1 = nn.Sigmoid()
        self.pool1d2 = nn.AvgPool1d(2)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(lstm_units * 2, lstm_units*2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(lstm_units, 200)
        self.act4 = nn.Sigmoid()
    def forward(self, x):
        # x = x.transpose(-1, -2)
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.pool1d2(x)  # bs, lstm_units, 1
        x = self.drop(x)
        # x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        # x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units
        attn = self.act3(attn)
        x = x * attn
        x = global_average_pooling(x)
        x = self.cls(x)
        x = self.act4(x)
        return x

    # def __init__(self, window=60, dim=200, lstm_units=57, num_layers=2):
    #     super(CNNLSTMModel_ECA, self).__init__()
    #     self.conv1d = nn.Conv1d(dim, lstm_units, 3,padding=1)
    #     self.act1 = nn.Sigmoid()
    #     self.pool1d = nn.AvgPool1d(2)
    #     self.drop = nn.Dropout(p=0.01)
    #     self.lstm = nn.LSTM(lstm_units, lstm_units * 2, batch_first=True, num_layers=1, bidirectional=True)
    #     self.act2 = nn.Tanh()
    #     self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
    #     self.act3 = nn.Sigmoid()
    #     self.cls = nn.Linear(lstm_units * 2, 200)
    #     self.act4 = nn.Tanh()

    # def forward(self, x):
    #     # x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
    #     x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
    #     x = self.act1(x)
    #     x = self.pool1d(x)  # bs, lstm_units, 1
    #     x = self.drop(x)
    #     # x = x.transpose(-1, -2)  # bs, 1, lstm_units
    #     x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
    #     x = self.act2(x)
    #
    #     x = x.squeeze(dim=1)  # bs, 2*lstm_units
    #     attn = self.attn(x)  # bs, 2*lstm_units
    #     attn = self.act3(attn)
    #     x = x * attn
    #     x = global_average_pooling(x)
    #     x = self.cls(x)
    #     x = self.act4(x)
    #     return x


def init_rnn(x, type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif type == 'uniform':
                    stdv = 1.0 / (getattr(x, w).size(-1)) ** 0.5
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif type == 'normal':
                    stdv = 1.0 / (getattr(x, w).size(-1)) ** 0.5
                    init.normal_(getattr(x, w), 0.0, stdv)
                else:
                    raise ValueError


class Transformer(nn.Module):
    def __init__(self, feature_num, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(feature_num, d_model)
        self.tf1 = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.5)
        self.tf2 = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(d_model, 200)

    def forward(self, x):
        x = self.embedding(x)
        x = self.tf1.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.dropout(x)
        x = self.tf2.encoder(x)
        x = self.decoder(x)

        return x


# Modified from https://github.com/dharwath/DAVEnet-pytorch.git
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x

class DavenetSmall(nn.Module):
    def __init__(self, input_dim, embedding_dim=1024):
        super(DavenetSmall, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,2))
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,3), stride=(1,1), padding=(0,2))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.squeeze(2)
        return x

class SentenceRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SentenceRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=40, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.size(0)
        T = x.size(1)
        if torch.cuda.is_available():
          h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
          c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
        
        embed, _ = self.rnn(x, (h0, c0))
        print('embed.size(): ', embed.size())
        out = embed[:, :, :self.embedding_dim] + embed[:, :, self.embedding_dim:]
        return out

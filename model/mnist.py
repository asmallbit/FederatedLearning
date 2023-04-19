import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2)   # 添加Dropout层
        )

        self.conv2 = nn.Sequential(     
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2)   # 添加Dropout层
        )

        self.fc1 = nn.Linear(7*7*32,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*32)
        output = F.relu(self.fc1(output))
        output = F.dropout(output, p=0.5, training=self.training)  # 添加Dropout层
        output = self.fc2(output)
        return F.log_softmax(output,dim=1)


class MNIST_RNN(nn.Module):
    def __init__(self):
        super(MNIST_RNN, self).__init__()

        self.rnn = nn.Sequential(
            nn.LSTM(         
                input_size=28,
                hidden_size=128,         
                num_layers=1,           
                batch_first=True,       
            ),
            nn.Dropout(0.5)
        )

        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1,28,28)
        r_out, (h_n, h_c) = self.rnn(x, None)   

        out = self.out(r_out[:, -1, :])
        return out
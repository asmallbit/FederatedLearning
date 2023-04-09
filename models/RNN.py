import torch.nn as nn

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

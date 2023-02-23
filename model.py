import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, x_size , n_channels=10,dropout_p=0.1) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, n_channels, kernel_size=x_size,bias=False),
            nn.PReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.linear = nn.Sequential(
            nn.Linear(n_channels, 5),
            nn.PReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(5,1)
        )

    def forward(self,x):
        x = self.cnn(x)
        x = x.view(-1,self.n_channels)
        x = self.linear(x)
        return x

def build_model(**kwargs):
    x_size = kwargs["x_size"]

    if "n_channels" not in kwargs:
        n_channels = 10
    else:
        n_channels = kwargs["n_channels"]

    if "dropout_p" not in kwargs:
        dropout_p= 0.1
    else:
        dropout_p = kwargs["dropout_p"]

    return MyModel(x_size=x_size,n_channels=n_channels,dropout_p=dropout_p)
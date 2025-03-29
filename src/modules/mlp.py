from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        drop=0.0,
        activation=nn.GELU,
        bias=True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

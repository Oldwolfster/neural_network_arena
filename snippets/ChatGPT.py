class DenseLayer(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.layer = None

    def initialize(self, in_features: int):
        self.layer = nn.Linear(in_features, self.out_features)

    def forward(self, x):
        if self.layer is None:
            self.initialize(x.size(1))  # Lazy initialization based on input
        return self.layer(x)


class ChatGPTisKing(nn.Module):
    def __init__(self, *layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(out) for out in layers])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ChatGPTisKing(2, 4, 1)  # Input inferred

    def forward(self, x):
        return self.net(x)

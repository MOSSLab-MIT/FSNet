import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(MLP, self).__init__()
        # layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(p=dropout)]
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(p=dropout/(i+1))]
            # layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(p=dropout/(3*i+1))]
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        # Initialize weights when the model is created
        # for m in self.mlp:
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_uniform_(m.weight)
        #             if m.bias is not None:
        #                 nn.init.zeros_(m.bias)
               
    def forward(self, x):
        return self.mlp(x)



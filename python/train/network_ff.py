import torch
from torch import nn

HIDDEN_SIZE = 512


class FeedForwardNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(FeedForwardNN, self).__init__()

        # Reads only the first `input_size` floats, so a network built for
        # the legacy 2016-float state still runs on the current one.
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state[..., : self.input_size]))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

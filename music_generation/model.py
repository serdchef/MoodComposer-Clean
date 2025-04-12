
import torch.nn as nn

class NoteLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(NoteLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

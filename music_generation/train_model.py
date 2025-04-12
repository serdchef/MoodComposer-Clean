# music_generation/train_model.py

import torch
import torch.nn as nn
import numpy as np
from data import get_note_sequence
from music_generation.model import NoteLSTM

input_size = 1
hidden_size = 128
num_layers = 2
output_size = 1
seq_length = 4
epochs = 300
lr = 0.01

notes = get_note_sequence()
X, y = [], []
for i in range(len(notes) - seq_length):
    X.append(notes[i:i+seq_length])
    y.append(notes[i+seq_length])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoteLSTM(input_size, hidden_size, num_layers, output_size).to(device)
X, y = X.to(device), y.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    output = model(X)
    loss = criterion(output.squeeze(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")

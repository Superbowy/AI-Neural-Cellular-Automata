import time

import numpy as np
import torch
import torch.nn as nn

import utils
from config import DEVICE, EMOJI
from NeuralCellularAutomata import NeuralCellularAutomata
from Pattern import Pattern


def train(model: NeuralCellularAutomata, pt: Pattern, epochs, batch_size, lr=0.002):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    start_time = time.time()
    target = torch.tensor(pt.data, device=DEVICE, dtype=torch.float32)
    target = target.expand(batch_size, -1, -1, -1)
    init_states = torch.zeros((batch_size, 16, 100, 100)).to(DEVICE)
    init_states[:, 3:, 50, 50] = torch.ones(13)

    for epoch in range(epochs):
        model.train()
        y_pred = model.forward(init_states, np.random.randint(64, 97))[:, -1, :, :, :]
        input = y_pred[:, :4, :, :]
        loss = loss_fn(input, target)
        optimizer.zero_grad()
        loss.backward()
        utils.normalize_grads(
            model
        )  # -----------> Absolument primordial - stagne sans (ou bcp plus long + incertain)
        optimizer.step()

        if (epoch + 1) % (epochs / 10) == 0:
            with torch.no_grad():
                print(
                    f"Epoch : {epoch + 1}. Loss : {loss:.3f}. Speed : {(epochs/10) / (time.time() - start_time):.2f} epoch/s"
                )
            start_time = time.time()


NCA0 = NeuralCellularAutomata().to(DEVICE)
PT0 = Pattern(EMOJI)

train(NCA0, PT0, 150, 8)
utils.generate_gif(NCA0)

import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import utils
from config import DEVICE, EMOJI, GRID_SIZE
from NeuralCellularAutomata import NeuralCellularAutomata
from Pattern import Pattern

# Reminder : shape is N, B, C, H, W


def train(model: NeuralCellularAutomata, pt: Pattern, epochs, batch_size, lr):
    
    print(f"[!] Training with {DEVICE}")

    loss_fn = nn.MSELoss(reduction="none") # Reduction none so we can replace the correct outputs later on
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    start_time = time.time()
    target = torch.tensor(pt.data, device=DEVICE, dtype=torch.float32)
    target = target.unsqueeze(0).expand(batch_size, -1, -1, -1)

    init = torch.zeros(
        (16, GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32
    )
    init[3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1

    pool = init.repeat(1024, 1, 1, 1)


    for epoch in range(epochs + 1):
        model.train()

        idxs = np.random.randint(0, 1024, batch_size) 
        batch = pool[idxs]

        batch[-3:] *= utils.perturb(3).to(DEVICE)

        y_pred = model.forward(batch, np.random.randint(64, 96))
        input = y_pred[:, :4, :, :]
        loss = loss_fn(input, target)
        loss_per_batch = loss.mean(dim=(1, 2, 3))
        loss = loss.mean()
        loss.backward()
        utils.normalize_grads(model)
        optimizer.step()
        optimizer.zero_grad()

        results = y_pred.detach()
        results[torch.argmax(loss_per_batch)] = init
        pool[idxs] = results

        if (epoch) % (epochs / 10) == 0:
            print(
                f"Epoch : {epoch}. Loss : {loss:.3f}. Speed : {(epochs/10) / (time.time() - start_time):.2f} epochs/s"
            )
            start_time = time.time()

def run():
    NCA0 = NeuralCellularAutomata().to(DEVICE)
    PT0 = Pattern(EMOJI)

    BATCH_SIZE = 8
    EPOCHS = 1000
    LR = 0.001

    train(NCA0, PT0, EPOCHS, BATCH_SIZE, LR)

    torch.save(NCA0.state_dict(), f"NCA0_{EMOJI}_{EPOCHS}.pth")

    input_states = torch.zeros(
        (1, 16, GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32
    )
    input_states[:, 3:, GRID_SIZE // 2 , GRID_SIZE // 2] = 1
    utils.generate_gif(NCA0, input_states, steps=200)


if __name__ == "__main__":
    run()

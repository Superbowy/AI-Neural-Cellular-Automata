import numpy as np
import torch
from torch import nn

import utils


class NeuralCellularAutomata(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(
                16, 16 * 3, 3, groups=16, padding=1, bias=False
            ),  # Groups = 16 to treat all channels seperately # Why 3 ? Why not 3 * 3 ?
            nn.Conv2d(16 * 3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 16, 1),
        )
        # A commenter si pb
        self.sequence[-1].weight.data *= 0
        # nn.init.normal_(self.sequence[-1].weight, mean=0, std=0.01)

        id = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8  # Normalizing
        full_kernel = np.stack([id, sobel_x, sobel_x.T], axis=0)
        full_kernel = np.tile(full_kernel, [16, 1, 1])

        self.sequence[0].weight.data = torch.tensor(
            full_kernel, dtype=torch.float32
        ).unsqueeze(
            1
        )  # Addind dimension to match (N, C, H, W)
        self.sequence[0].weight.requires_grad = False

    def forward(self, input_states: torch.Tensor, steps, return_all_frames=False):
        batch_size = input_states.shape[0]
        current_state = input_states
        if return_all_frames:
            frames = [current_state]

        for _ in range(steps):
            next_state = current_state + self.sequence(
                current_state
            ) * utils.living_mask(current_state)# * utils.bernouilli_mask(batch_size)
            current_state = next_state
            if return_all_frames:
                frames.append(next_state)

        return torch.stack(frames) if return_all_frames else current_state



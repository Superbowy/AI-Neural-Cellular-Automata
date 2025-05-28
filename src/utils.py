from NeuralCellularAutomata import NeuralCellularAutomata
from config import DEVICE
import numpy as np
import torch
import PIL.Image as Image

def bernouilli_mask(batch_size, p = 0.9):
    mask = np.random.rand(1, 16, 100, 100) < p
    return torch.tensor(mask, device=DEVICE, dtype=torch.float32)

def normalize_grads(model):
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad


def generate_gif(model : NeuralCellularAutomata):
    with torch.inference_mode():
        init_states = torch.zeros((1, 16, 100, 100)).to(DEVICE)
        init_states[:, 3:, 50, 50] = torch.ones(13)
        frames = model.forward(init_states, 80)

    split = 1
    frames = frames[::split]

    images = []
    for frame in frames:
        img = np.transpose(frame[0, :4, :, :].cpu().numpy(), (2, 1, 0))
        img = np.clip(img, 0, 1)

        img = (img[:, :, :3] * 255).astype(np.uint8)
        images.append(Image.fromarray(img))

    images[0].save(
        'nca_evolution.gif',
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )

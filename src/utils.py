from config import DEVICE, GRID_SIZE
import numpy as np
import torch.nn.functional as F
import torch
import PIL.Image as Image

def living_mask(input):
    channel_3 = (input[:, 3:4] > 0.1).float()
    kernel = torch.ones(1, 1, 3, 3, device=input.device)
    neighbor_count = F.conv2d(channel_3, kernel, padding=1)
    alive = (neighbor_count > 0).float()
    return alive.repeat(1, 16, 1, 1)

def bernouilli_mask(batch_size, p = 0.9):
    return (torch.rand(batch_size, 16, GRID_SIZE, GRID_SIZE, device=DEVICE, dtype=torch.float32) < p)

def normalize_grads(model):
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

def perturb(n_perturb):
    x = torch.linspace(-1.0, 1.0, GRID_SIZE)[None, None, :]
    y = torch.linspace(-1.0, 1.0, GRID_SIZE)[None, :, None]

    radius = 2 * (GRID_SIZE / 16) / GRID_SIZE
    offset_range = 2 * (GRID_SIZE / 8) / GRID_SIZE

    center_offsets = (torch.rand(n_perturb, 2, 1, 1) * 2 - 1) * offset_range
    center_x = center_offsets[:, 0]
    center_y = center_offsets[:, 1]

    x_norm = (x - center_x) / radius
    y_norm = (y - center_y) / radius

    mask = 1.0 - ((x_norm ** 2 + y_norm ** 2) < 1.0).float()

    mask = mask.unsqueeze(1).repeat(1, 16, 1, 1)

    return mask


def generate_gif(model, input_states : torch.Tensor, steps=80):
    with torch.inference_mode():
        frames = model.forward(input_states, steps, return_all_frames=True)

    images = []
    for frame in frames:
        img = np.transpose(frame[0, :4, :, :].cpu().numpy(), (2, 1, 0))
        img = np.clip(img, 0, 1)

        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img, "RGBA"))

    images[0].save(
        'nca_evolution.gif',
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )

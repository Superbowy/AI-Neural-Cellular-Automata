import numpy as np
from math import floor
import time
import pygame
import torch

from config import DEVICE, GRID_SIZE
from NeuralCellularAutomata import NeuralCellularAutomata

# Config
DISPLAY_SCALE = 10
FPS = 10

# Model setup
NCA0 = NeuralCellularAutomata().to(DEVICE)
NCA0.load_state_dict(torch.load("../NCA0_mage_2000.pth", map_location=DEVICE))
NCA0.eval()

# Init grid
state = torch.zeros((1, 16, GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=DEVICE)
state[:, :, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0

# Init pygame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * DISPLAY_SCALE, GRID_SIZE * DISPLAY_SCALE))
clock = pygame.time.Clock()

paused = False
running = True

def zero_circle(grid, center_x, center_y):
    radius = GRID_SIZE / 16
    _, C, H, W = grid.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=DEVICE),
        torch.arange(W, device=DEVICE),
        indexing="ij"
    )
    mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius ** 2  # (H, W)
    grid[0, :, mask] = 0.0
    grid[0, 3, mask] = 1.0

while running:
    if not paused:
        with torch.no_grad():
            state = NCA0(state, 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            paused = not paused
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            gx = floor(mx / DISPLAY_SCALE)
            gy =floor(my / DISPLAY_SCALE)
            zero_circle(state, gy, gx) #-> gy and gy are reversed because we transpose after

    rgba = state[0, :4].clamp(0, 1).cpu().permute(2, 1, 0).numpy()
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    rgba_bytes = rgba_uint8.tobytes()
    surface = pygame.image.frombuffer(rgba_bytes, (GRID_SIZE, GRID_SIZE), "RGBA")

    surface = pygame.transform.scale(
        surface, (GRID_SIZE * DISPLAY_SCALE, GRID_SIZE * DISPLAY_SCALE)
    )
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    clock.tick(FPS)
pygame.quit()

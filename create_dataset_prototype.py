import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from perlin_noise import PerlinNoise

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '..', 'Code_Permeability'))
sys.path.append(target_dir)
import flow_toolbox as flow

def main():
    seed = 2001
    dim = (N, M) = (400, 400)

    noise1 = PerlinNoise(octaves=5, seed=seed)
    noise2 = PerlinNoise(octaves=10, seed=seed)

    K = np.zeros(dim, dtype=np.double)
    for i in range(N):
        for j in range(M):
            K[i, j] += 2 / 3 * noise1([i / (N - 1), j / (M - 1)])
            K[i, j] += 1 / 3 * noise2([i / (N - 1), j / (M - 1)])

    K[:, 0] *= 0
    K[:, -1] *= 0
    K *= 10
    K *= K
    K += .001
    perm = flow.create_continuous_permeability(K)
    corners = (100, 100, 0, 0)
    boundaries = flow.create_boundary_conditions(perm, corners)
    pressure = flow.better_solve_for_pressure(perm, boundaries)

    u, v = perm.compute_uv(pressure)
    gu = flow.on_cell(u, 0)
    gv = flow.on_cell(v, 1)
    speed = np.sqrt(gu * gu + gv * gv)

    plt.figure()
    plt.imshow(K, cmap='viridis', interpolation='nearest')
    plt.figure()
    plt.imshow(boundaries, cmap='viridis', interpolation='nearest')
    plt.figure()
    plt.imshow(speed, cmap='viridis', interpolation='nearest')
    plt.contour(K, 5)
    plt.show()
    

if __name__ == "__main__":
    main()
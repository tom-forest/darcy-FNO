import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from perlin_noise import PerlinNoise
import multiprocessing

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '..', 'Code_Permeability'))
sys.path.append(target_dir)
import flow_toolbox as flow

def create_sq_perlin_noise(octaves, intensities, shape, value_range, seed=None):
    ''' creates a random matrix using squared Perlin noise
        permeability range is a tuple of form (min permability, max permeability)'''

    nf = len(octaves)
    assert len(intensities) == nf

    intensities = np.array(intensities)

    noise = []
    for i in range(nf):
        noise.append(PerlinNoise(octaves=octaves[i], seed=seed))

    K = np.zeros(shape, dtype=np.double)
    N, M = shape
    for i in range(N):
        for j in range(M):
            for k in range(nf):
                K[i, j] += intensities[k] * noise[k]([i / (N - 1), j / (M - 1)])

    min_value, max_value = value_range

    K /= np.sum(intensities)
    K *= K
    #K = 1 - K
    minK = np.min(K)
    maxK = np.max(K)
    K -= minK
    K /= maxK - minK
    K *= max_value - min_value
    K += min_value

    return K

def create_training_sample(shape, permeability_range, boundary_pressure_range, seed):
    ''' creates one training sample'''

    np.random.seed(seed)

    # perlin noise settings - could be put elsewhere
    max_n_octave = 6 # max number of octaves to use for the perlin noise, exclusive
    min_n_octave = 1 # min //, inclusive
    min_octave = 1 # minimum octave to use, inclusive
    max_octave = 20 # maximum //, exclusive

    # create permeability grid
    n_octave = np.random.randint(min_n_octave, max_n_octave)
    octaves = np.random.uniform(min_octave, max_octave, size=n_octave)
    intensities = 1 / octaves

    K = create_sq_perlin_noise(octaves, intensities, shape, permeability_range, seed)
    permeability = flow.create_continuous_permeability(K)

    # create boundary pressure conditions
    min_p, max_p = boundary_pressure_range
    corners = np.random.uniform(min_p, max_p, size=4)

    boundary_conditions = flow.create_boundary_conditions(permeability, corners)

    # solve Darcy's flow equation for these parameters
    pressure = flow.better_solve_for_pressure(permeability, boundary_conditions)

    return permeability.K2D, boundary_conditions, pressure

def sample_creation_worker(shape, permeability_range, boundary_pressure_range, sample_queue, seed):
    ''' adds one new sample to the provided queue. Intended for use with multiprocessing.'''

    perm, bound, pressure = create_training_sample(shape, permeability_range, boundary_pressure_range, seed)
    x = np.array([perm, bound])
    y = np.array([pressure])
    sample_queue.put((x, y))

def create_samples_multiproc(n_samples, shape, permeability_range, boundary_pressure_range, seed):
    ''' creates n_samples training samples using multiprocessing. CAUTION: high ram usage for large n_sample'''

    np.random.seed(seed)
    seed_array = np.random.randint(1, 100000, size=n_samples)

    processes = []
    sample_queue = multiprocessing.Manager().Queue()

    for i in range(n_samples):
        process = multiprocessing.Process(target=sample_creation_worker, args=(shape, permeability_range, boundary_pressure_range, sample_queue, int(seed_array[i])))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    samples = []
    while not sample_queue.empty():
        samples.append(sample_queue.get())

    return samples

def view_darcy(samples):
    ''' plots visualizations of provided data'''

    for sample in samples:
        x, y = sample

        perm = x[0]
        bound = x[1]
        pressure = y[0]

        K = perm[1:-1, 1:-1]
        permeability = flow.create_continuous_permeability(K)

        u, v = permeability.compute_uv(pressure)
        gu = flow.on_cell(u, 0)
        gv = flow.on_cell(v, 1)
        speed = np.sqrt(gu * gu + gv * gv)

        pressure = pressure[1:-1, 1:-1]

        plt.figure()
        plt.imshow(K, cmap='plasma', interpolation='nearest')
        plt.figure()
        plt.imshow(bound, cmap='viridis', interpolation='nearest')
        plt.figure()
        plt.imshow(pressure, cmap='viridis', interpolation='nearest')
        plt.contour(K, 5, cmap='plasma')
        plt.quiver(gv, gu, pivot='mid', angles='xy')
        plt.show()

def main():
    visualize = True

    n_samples = 1
    seed = 1
    dim = (N, M) = (100, 100)
    perm_range = (0.001, 100)
    boundary_p_range = (0, 100)

    samples = create_samples_multiproc(n_samples, dim, perm_range, boundary_p_range, seed)
    
    if visualize:
        view_darcy(samples)

if __name__ == "__main__":
    main()
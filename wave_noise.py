import numpy as np
import matplotlib.pyplot as plt
''' Module to generate pure sinusoidal 2d noise'''


default_energy_p = 2

def p_energy_spectrum(f, p):
    ''' Simple energy spectrum function (f+1)^(-p)'''
    return np.float_power(f, -p)


class Wave_noise:
    ''' Wave noise generator'''

    def __init__(self, frequencies, energy_spectrum=lambda f:p_energy_spectrum(f, default_energy_p), seed = None):
        # store frequencies
        self.freq = np.array(frequencies)
        self.n_freq = len(frequencies)

        # compute energy sectrum
        self.spec = energy_spectrum(self.freq)
        self.max_amp = np.sum(self.spec)

        # set random generator
        self.rng = np.random.default_rng(seed)

    
    def __call__(self, shape):
        ''' Returns wave noise evaluated on a grid of the provided shape, with x and y values in [0,1]'''

        # generate wavefront normals
        normal_angles = self.rng.uniform(0, 2 * np.pi, self.n_freq)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        normal = np.array([nx, ny])

        # generate phases
        phase = self.rng.uniform(0, 2 * np.pi, self.n_freq)

        # generate coordinate matrix
        lx = np.linspace(0, 1, shape[0], endpoint=True)
        ly = np.linspace(0, 1, shape[1], endpoint=True)
        x, y = np.meshgrid(lx, ly)
        coord = np.array([x, y])

        # project coordinates of each point on each normal
        coord = np.einsum('ijk,ih->hjk', coord, normal)

        # apply frequencies
        coord = np.einsum('ijk,i->ijk', coord, self.freq)
        coord *= np.pi

        # add phases
        coord += np.reshape(phase, (-1, 1, 1))

        # compute wave
        coord = np.sin(coord)

        # apply magnitudes and sum
        coord = np.einsum('ijk,i->jk', coord, self.spec)

        # divide by total magnitude
        coord /= self.max_amp

        return coord


def main():
    seed = 1

    freq = []
    for i in range(40):
        freq.append(pow(2, i+2))
    a = Wave_noise(freq, seed=seed)

    for i in range(3):
        b = a((1000, 1000))

        plt.imshow(b)
        plt.show()

if __name__ == "__main__":
    main()
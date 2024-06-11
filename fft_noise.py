import numpy as np
import matplotlib.pyplot as plt


def default_spectrum(coords, p):
    ''' returns the default fourier filter magnitude = frequency ^-p of the provided frequencies'''

    norms = np.linalg.norm(coords, axis=0)
    return np.float_power(norms, -p)


class fft_noise:

    def __init__(self, magnitude_map=lambda c:default_spectrum(c,2), seed=None):
        self.mag = magnitude_map
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        ''' Returns noise evaluated on a grid of the provided shape'''

        # generate signals
        amp = self.rng.uniform(0, 1, shape)
        phase = self.rng.uniform(0, 2 * np.pi, shape)
        signal = amp * np.exp(1.j * phase)

        # compute filter
        lx = np.linspace(-.5, .5, shape[0], endpoint=True)
        ly = np.linspace(-.5, .5, shape[1], endpoint=True)
        X, Y = np.meshgrid(lx, ly)
        coord = np.array([X, Y])
        filter = self.mag(coord)

        # apply filter
        fourier_noise = filter * signal
        fourier_noise = np.fft.ifftshift(fourier_noise)
        noise = np.fft.ifft2(fourier_noise)

        return np.real(noise)
    
    
def main():
    a = fft_noise(seed=1)
    b = a((1000, 1000))
    plt.imshow(b)
    plt.show()

if __name__ == "__main__":
    main()
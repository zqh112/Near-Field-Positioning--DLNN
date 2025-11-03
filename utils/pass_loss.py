import numpy as np

def calculate_path_loss(r: float, f: float, G_t: float=10**1.5, G_r: float=10**0.5):
    '''
    description:
    param {float} r: length of the central link
    param {float} f: carrier frequency
    param {float} G_t: transmit antenna gain
    param {float} G_r: receive antenna gain
    return {float} path_loss: a.k.a. channel gain
    '''
    wavelength = 3e8 * (1  / f)
    path_loss = ((wavelength / (4 * np.pi))**2) * (G_r * G_t / r**2)
    return path_loss

if __name__ == '__main__':
    main()
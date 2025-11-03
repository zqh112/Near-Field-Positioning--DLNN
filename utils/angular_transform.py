import numpy as np
from .ULA import ULA

def angular_transform(H: np.ndarray, ULA_BS: ULA, f=28e9):
    '''
    description: transform H from antenna space to angular space
    param {np.ndarray} H
    param {ULA} ULA_BS
    param {float} f: carrier freq (default as 28 GHz)
    return {tuple} H_a (angular domain channel), A (WTM), epsilon_arr (index)
    '''
    M = ULA_BS.num_of_antennas
    D = ULA_BS.aperture
    wave_length = 3e8 / f
    k = 2 * np.pi / wave_length
    epsilon_arr = np.arange(start=np.ceil(-D / wave_length), stop=np.floor(D / wave_length)+1, step=1, dtype=int)
    A = np.hstack([np.array(
        (1 / np.sqrt(M)) * np.exp(1j * ((2 * np.pi / D) * epsilon * ULA_BS.element_coordinate[0, :])), ndmin=2).T
                     for epsilon in epsilon_arr])
    H_a = A.T.conjugate() @ H @ A
    return H_a, A, epsilon_arr

if __name__ == '__main__':
    main()
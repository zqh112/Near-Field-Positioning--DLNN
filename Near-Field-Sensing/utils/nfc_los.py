import numpy as np
from .ULA import ULA

def generate_nfc_los_channels(ULA_BS: ULA, target_coordinate: np.ndarray, f: float = 28e9):
    """
    Function: calculate NFC LoS channel matrix
    :param ULA_BS: ULA obj.
    :param f: float, carrier frequency
    :param target_coordinate: the position of the target.
    :return: H: nd.array in shape [ULA_Tx.num_of_antennas, ULA_Rx.num_of_antennas]
    """
    H = np.zeros((), dtype=np.complex64)  # overall channel

    h = np.exp(-1j * (2 * np.pi / (3e8 / f)) * np.linalg.norm((ULA_BS.element_coordinate.T -
                                                               target_coordinate), ord=2, axis=-1)).reshape(-1, 1)

    H = h @ h.transpose()
    return H

if __name__ == '__main__':
    main()
import numpy as np
from matplotlib import pyplot as plt
from utils.ULA import ULA
from utils.angular_transform import angular_transform
from utils.nfc_los import generate_nfc_los_channels
from utils.pass_loss import calculate_path_loss

def music_algorithm(observation: np.ndarray, ULA_BS: ULA, dist_range: list,
                    angle_range: list, num_of_grid: int, visualization: bool=False):
    f = 28e9
    R = observation @ observation.conj().T
    eigenvalues, eigenvectors = np.linalg.eig(R) # eigenvalue decomposition
    idx = eigenvalues.argsort()[::-1] # descending order
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    D_s = sorted_eigenvalues[0]
    E_s = sorted_eigenvectors[:, 0]
    E_n = sorted_eigenvectors[:, 1:]
    U = E_n @ E_n.conj().T
    if not visualization:
        # for a fair comparison with the NN method, we only sample the grids in the region where the training data is generated.
        dist_arr=np.linspace(dist_range[0], dist_range[1], num=int(np.sqrt(num_of_grid)))
        angle_arr=np.linspace(angle_range[0], angle_range[1], num=int(np.sqrt(num_of_grid)))
        proj_min = 10e3
        target_position = np.zeros(2)
        for dist in dist_arr:
            for angle in angle_arr:
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                a = np.exp(-1j * (2 * np.pi / (3e8 / f)) * np.linalg.norm((ULA_BS.element_coordinate.T - target_coordinate),
                                                                        ord=2, axis=-1)).reshape(-1, 1)
                proj = np.real(a.conj().T @ U @ a)
                if proj < proj_min:
                    proj_min = proj
                    target_position[0] = dist * np.cos(angle)
                    target_position[1] = dist * np.sin(angle)
        return target_position
    else:
        # for a fair comparison with the NN method, we only sample the grids in the region where the training data is generated.
        dist_arr=np.linspace(dist_range[0], dist_range[1], num=num_of_grid)
        angle_arr=np.linspace(angle_range[0], angle_range[1], num=num_of_grid)
        # spectrum = np.zeros((dist_arr.shape[0], angle_arr.shape[0]))
        proj_min = 10e3
        target_position = np.zeros(2)
        for dist_idx, dist in enumerate(dist_arr):
            for angle_idx, angle in enumerate(angle_arr):
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                a = np.exp(-1j * (2 * np.pi / (3e8 / f)) * np.linalg.norm((ULA_BS.element_coordinate.T - target_coordinate),
                                                                        ord=2, axis=-1)).reshape(-1, 1)
                proj = np.real(a.conj().T @ U @ a)
                # spectrum[dist_idx, angle_idx] = proj
                if proj < proj_min:
                    proj_min = proj
                    target_position[0] = dist * np.cos(angle)
                    target_position[1] = dist * np.sin(angle)
        return target_position

def sense_one_round_music(ULA_BS: ULA, H: np.ndarray, A: np.ndarray, dist_range: list,
                    angle_range: list, transmit_power: float, noise_power: float, num_of_grid: int,
                    visualization: bool=False):
    M = H.shape[0]
    # To obtain the Rx observation:
    # 1. transmit two DL pilots
    noise = np.sqrt(0.5 * noise_power) * np.random.normal(0, 1, (M, 1)) + 1j * np.sqrt(0.5 * noise_power) * np.random.normal(0, 1, (M, 1))
    beamformer = np.sqrt(transmit_power) * (A @  np.ones((M, 1))) / np.linalg.norm((A @  np.ones((M, 1))).flatten(), ord=2)
    observation = H @ beamformer + noise
    res = music_algorithm(observation=observation, ULA_BS=ULA_BS, dist_range=dist_range,
                          angle_range=angle_range, num_of_grid=num_of_grid,
                          visualization=visualization)
    if not visualization:
        target_position = res
        return target_position
    else:
        target_position = res
        return target_position
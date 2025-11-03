import numpy as np
import torch
import os
from torch.multiprocessing import Pool
import itertools as it
import sys
sys.path.append('/data/zqh/code/Near-Field-Sensing/')
from tqdm import tqdm
from utils.ULA import *
from utils.angular_transform import *
from utils.nfc_los import generate_nfc_los_channels


def generate_sample_helper(args):
    '''
    description: generates a single data point for the dataset.
    param {int} idx
    param {float} dist
    param {float} angle
    param {ULA} ULA_BS
    param {float} Carrier
    return {tuple} (label_xz, observation)
    '''
    idx, dist, angle, ULA_BS, f = args
    x = dist * np.cos(angle)
    z = dist * np.sin(angle)
    target_coordinate = np.array([x, 0.0, z])
    H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
    H_a, _, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
    # observation = torch.from_numpy(np.abs(H_a @ np.ones((ULA_BS.num_of_antennas, 1)))).flatten().float()
    observation = np.abs(H_a @ np.ones((ULA_BS.num_of_antennas, 1))).flatten()

    # return (torch.tensor([x, z]), observation)
    return (np.array([x, z], dtype=np.float32), observation.astype(np.float32))


def generate_dataset_for_CNN(dist_arr: np.ndarray, angle_arr: np.ndarray, ULA_BS: ULA,
                             f: float, num_of_processes: int, dataset_path: str = '/data/zqh/code/Near-Field-Sensing/samples/'):
    '''
    description: generate dataset for training CNN (or DNN)
    param {np.ndarray} dist_arr
    param {np.ndarray} angle_arr
    param {ULA} ULA_BS: ULA
    param {float} f: carrier freq
    param {int} num_of_processes: number of processes
    param {str} dataset_path: where the dataset is saved.
    return {*}
    '''
    os.makedirs(dataset_path, exist_ok=True)
    label_polar_arr = np.array(list(it.product(dist_arr, angle_arr)))
    args_list = [(idx, dist, angle, ULA_BS, f) for idx, (dist, angle) in enumerate(label_polar_arr)]
    with Pool(processes=num_of_processes) as pool:
        results = list(tqdm(pool.imap(generate_sample_helper, args_list),
                            total=len(args_list), desc='Generating data'))

    label_xz_arr, observation_arr = zip(*results)
    # label_xz_arr = torch.stack(label_xz_arr)
    # observation_arr = torch.stack(observation_arr)
    label_xz_arr = torch.tensor(np.stack(label_xz_arr), dtype=torch.float32)
    observation_arr = torch.tensor(np.stack(observation_arr), dtype=torch.float32)
    dataset_path = os.path.join(dataset_path, f'CNN-{int(ULA_BS.num_of_antennas)}')
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(label_xz_arr, os.path.join(dataset_path, 'label.pth'))
    torch.save(observation_arr, os.path.join(dataset_path, 'observation_origi_arr.pth'))

if __name__ == '__main__':
    M = 511 # num of antennas
    f = 28e9 # carrier freq. 28 GHz
    wave_length = 3e8 / f # calculate wavelength
    d = wave_length / 2 # half-wavelength antennas
    ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))

    #* generate training dataset for CNN
    generate_dataset_for_CNN(dist_arr=np.arange(8, 35, step=0.01), angle_arr=np.arange(np.pi/4, 3* np.pi/4, step=0.01), ULA_BS=ULA_BS, num_of_processes=96,
                          f=f)
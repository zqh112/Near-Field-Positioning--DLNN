import time
import json
import numpy as np
import os
import torch
from music import sense_one_round_music
from utils.ULA import ULA
from utils.angular_transform import angular_transform
from utils.nfc_los import generate_nfc_los_channels
from utils.pass_loss import calculate_path_loss
from utils.sense_one_round import sense_one_round
from model.DLNN import DLNN

def test_model_and_benchmark_on_real_system(model, ULA_BS: ULA, target_dist_arr: np.ndarray,
                                            target_angle_arr: np.ndarray, dist_range: list,
                                            angle_range: list, transmit_power: float,
                                            noise_power: float, f: float, num_of_grid_arr: np.ndarray,
                                            data_path='/data/zqh/code/Near-Field-Sensing/figs/'):
    model.eval()
    compare_result_dict = dict()
    dlnn_rse_error_list = []
    dlnn_run_time_list = []
    with torch.no_grad():
        for dist in target_dist_arr:
            for angle in target_angle_arr:
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                path_loss = calculate_path_loss(r=2 * dist, f=f)
                H = np.sqrt(path_loss) * H
                time_start = time.time()
                preds = sense_one_round(model, H, A, transmit_power, noise_power).squeeze(0)
                time_end = time.time()
                dlnn_rse_error_list.append(np.sqrt((preds[0] - target_coordinate[0]) ** 2 +
                                                (preds[1] - target_coordinate[-1]) ** 2))
                dlnn_run_time_list.append(time_end - time_start)

        compare_result_dict['DLNN RMSE'] = np.round(np.mean(dlnn_rse_error_list), decimals=5)
        compare_result_dict['DLNN Avg. Run Time'] = np.round(np.mean(dlnn_run_time_list), decimals=5)

    for num_of_grid in num_of_grid_arr:
        music_rse_error_list = []
        music_run_time_list = []
        for dist in target_dist_arr:
            for angle in target_angle_arr:
                ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                path_loss = calculate_path_loss(r=2 * dist, f=f)
                H = np.sqrt(path_loss) * H
                # MUSIC method
                time_start = time.time()
                target_position_est = sense_one_round_music(ULA_BS=ULA_BS, H=H, A=A,
                                                                      transmit_power=transmit_power,
                                                                      noise_power=noise_power, num_of_grid=num_of_grid,
                                                                      angle_range=angle_range, dist_range=dist_range,
                                                                      visualization=True)

                time_end = time.time()
                music_rse_error_list.append(np.sqrt((target_position_est[0] - target_coordinate[0]) ** 2 +
                                                    (target_position_est[1] - target_coordinate[-1]) ** 2))
                music_run_time_list.append(time_end - time_start)
        compare_result_dict[f'MUSIC RMSE with Gird {num_of_grid ** 2}'] = np.round(np.mean(music_rse_error_list),
                                                                                   decimals=5)
        compare_result_dict[f'MUSIC Avg. Run Time with Grid {num_of_grid ** 2}'] = np.round(
            np.mean(music_run_time_list), decimals=5)

    print("=== Compare Results ===")
    for key, val in compare_result_dict.items():
        print(f"{key}: {val}")


def test_model_and_benchmark_dist_antenna_array(target_dist_arr: np.ndarray, target_angle_arr: np.ndarray,
                                                transmit_power: float, aperture_arr: list,
                                                noise_power: float, f: float, num_of_grid: int,
                                                dist_range: list, angle_range: list,
                                                data_path='/data/zqh/code/Near-Field-Sensing/figs/'):
    DLNN_res = dict()
    MUSIC_res = dict()
    for M in aperture_arr:
        model = DLNN()
        checkpoint = torch.load(os.path.join(f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}', 'DLNN'))
        model.load_state_dict(checkpoint)
        dlnn_rse_error_list = []
        dlnn_run_time_list = []
        model.eval()
        with torch.no_grad():
            for dist in target_dist_arr:
                for angle in target_angle_arr:
                    ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))
                    target_coordinate = np.array([dist * np.cos(angle), 0.0, dist * np.sin(angle)])
                    H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                    _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                    path_loss = calculate_path_loss(r=2 * dist, f=f)
                    H = np.sqrt(path_loss) * H
                    time_start = time.time()
                    preds = sense_one_round(model, H, A, transmit_power, noise_power).squeeze(0)
                    time_end = time.time()
                    dlnn_rse_error_list.append(np.sqrt((preds[0] - target_coordinate[0]) ** 2 +
                                                    (preds[1] - target_coordinate[-1]) ** 2))
                    dlnn_run_time_list.append(time_end - time_start)
                DLNN_res[f'DLNN with antenna {M} Accuracy dist {dist}'] = np.round(np.mean(dlnn_rse_error_list),
                                                                                 decimals=5)
                DLNN_res[f'DLNN with antenna {M} Run Time dist {dist}'] = np.round(np.mean(dlnn_run_time_list),
                                                                                 decimals=5)

        music_rse_error_list = []
        music_run_time_list = []
        for dist in target_dist_arr:
            for angle in target_angle_arr:
                ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))
                target_coordinate = np.array([dist * np.cos(angle), 0.0, dist * np.sin(angle)])
                H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                path_loss = calculate_path_loss(r=2 * dist, f=f)
                H = np.sqrt(path_loss) * H
                time_start = time.time()
                target_position_est = sense_one_round_music(ULA_BS=ULA_BS, H=H, A=A,
                                                                      transmit_power=transmit_power,
                                                                      noise_power=noise_power, num_of_grid=num_of_grid,
                                                                      angle_range=angle_range, dist_range=dist_range,
                                                                      visualization=True)

                time_end = time.time()
                music_rse_error_list.append(np.sqrt((target_position_est[0] - target_coordinate[0]) ** 2 +
                                                    (target_position_est[1] - target_coordinate[-1]) ** 2))
                music_run_time_list.append(time_end - time_start)
            MUSIC_res[f'MUSIC with antenna {M} Accuracy dist {dist}'] = np.around(np.mean(music_rse_error_list),
                                                                                  decimals=5)
            MUSIC_res[f'MUISC with antenna {M} Run Time dist {dist}'] = np.around(np.mean(music_run_time_list),
                                                                                  decimals=5)

    res = dict(DLNN_res, **MUSIC_res)
    print("=== Compare Results ===")
    for key, val in res.items():
        print(f"{key}: {val}")

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    M = 511  # num of antennas
    f = 28e9  # carrier freq. 28 GHz
    wave_length = 3e8 / f  # calculate wavelength
    d = wave_length / 2  # half-wavelength antennas
    ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))
    # angle_range = [np.pi / 4.0, 3.0 * np.pi / 4.0]
    # target_dist_arr = [10, 25]
    # dist_range = [10, 25]
    # target_angle_arr = np.random.uniform(angle_range[0], angle_range[1], 100)
    # test_model_and_benchmark_dist_antenna_array(target_dist_arr=target_dist_arr,
    #                                                   dist_range=dist_range,
    #                                                   angle_range=angle_range,
    #                                                   target_angle_arr=target_angle_arr,
    #                                                   transmit_power=10,
    #                                                   aperture_arr=np.array([255, 511]),
    #                                                   noise_power=10 ** (-10.5),
    #                                                   f=f, num_of_grid=100,
    #                                                   data_path='./figs')

    angle_range = [np.pi / 4, 3 * np.pi / 4]
    dist_range = [10.0, 20.0]
    target_dist_arr = np.random.uniform(dist_range[0], dist_range[1], 10)
    target_angle_arr = np.random.uniform(angle_range[0], angle_range[1], 10)
    model = DLNN()
    checkpoint = torch.load(os.path.join(f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}', 'DLNN'))
    model.load_state_dict(checkpoint)
    test_model_and_benchmark_on_real_system(model, ULA_BS=ULA_BS, target_dist_arr=target_dist_arr,
                                                                  target_angle_arr=target_angle_arr,
                                                                  dist_range=dist_range,
                                                                  angle_range=angle_range,
                                                                  transmit_power=0.01, noise_power=10 ** (-10.5), f=f,
                                                                  num_of_grid_arr=[10, 100, 1000])
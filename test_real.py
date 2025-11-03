import os
import torch
import scienceplots
from matplotlib import pyplot as plt
import numpy as np
from model.DLNN import DLNN
from utils.ULA import ULA
from utils.angular_transform import angular_transform
from utils.nfc_los import generate_nfc_los_channels
from utils.pass_loss import calculate_path_loss
from utils.sense_one_round import sense_one_round

def test_model_on_real_system(model, ULA_BS: ULA, target_dist_arr: np.ndarray,
                              target_angle_arr: np.ndarray, transmit_power_arr: np.ndarray,
                              noise_power: float, f: float, monte_carlo_times=10, fig_path='/data/zqh/code/Near-Field-Sensing/figs/'):
    model.eval()
    label_list = []
    preds_pow_1_list = []
    preds_pow_2_list = []
    with torch.no_grad():
        for dist in target_dist_arr:
            for angle in target_angle_arr:
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                path_loss = calculate_path_loss(r=2 * dist, f=f)
                H = np.sqrt(path_loss) * H
                preds = np.zeros((monte_carlo_times, 2))
                for it in range(monte_carlo_times):
                    preds[it] = sense_one_round(model, H, A, transmit_power_arr[0], noise_power)
                preds = np.mean(preds, axis=0)
                label_list.append(target_coordinate)
                preds_pow_1_list.append(preds)

        for dist in target_dist_arr:
            for angle in target_angle_arr:
                target_coordinate = np.array([dist * np.cos(angle), 0, dist * np.sin(angle)])
                H = generate_nfc_los_channels(ULA_BS=ULA_BS, target_coordinate=target_coordinate, f=f)
                _, A, _ = angular_transform(H=H, ULA_BS=ULA_BS, f=f)
                path_loss = calculate_path_loss(r=2 * dist, f=f)
                H = np.sqrt(path_loss) * H
                preds = np.zeros((monte_carlo_times, 2))
                for it in range(monte_carlo_times):
                    preds[it] = sense_one_round(model, H, A, transmit_power_arr[1], noise_power)
                preds = np.mean(preds, axis=0)
                label_list.append(target_coordinate)
                preds_pow_2_list.append(preds)

    label_arr = np.array(label_list)
    preds_pow_1_arr = np.array(preds_pow_1_list)
    preds_pow_2_arr = np.array(preds_pow_2_list)
    
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(2.4 * 1.618, 2.4))
        # ax.plot_trisurf
        ax.scatter(label_arr[:, 0], label_arr[:, -1], color='red', facecolors='none',
                   marker='s', s=10, label="Ground Truth")
        ax.scatter(preds_pow_1_arr[:, 0], preds_pow_1_arr[:, -1], color='blue', facecolors='none',
                   marker='o', s=10, label=f"DLNN, {int(10 * np.log10(transmit_power_arr[0] * 1e3))} dBm")
        ax.scatter(preds_pow_2_arr[:, 0], preds_pow_2_arr[:, -1], color='darkorange', facecolors='none',
                   marker='d', s=10, label=f"DLNN, {int(10 * np.log10(transmit_power_arr[1] * 1e3))} dBm")
        ax.set_xlabel(r'$x$', fontsize=11.5)
        ax.set_ylabel(r'$z$', fontsize=11.5)
        ax.legend(fontsize=9, frameon=True)
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, 'test_nn0801.pdf'))
        plt.show()

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    M = 511
    f = 28e9
    wave_length = 3e8 / f
    d = wave_length / 2
    ULA_BS = ULA(num_of_antennas=M, antenna_spacing=d, centre_coordinate=np.array([0, 0, 0]))
    model = DLNN()
    checkpoint = torch.load(os.path.join(f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}', 'DLNN'))
    model.load_state_dict(checkpoint)
    test_model_on_real_system(model=model, ULA_BS=ULA_BS,
                              target_dist_arr=[10, 15, 20],
                              target_angle_arr=[np.pi / 4, np.pi / 2, 3 * np.pi / 4 - 1e-8],
                              transmit_power_arr=np.array([10, 0.01]), noise_power=10 ** (-10.5),
                              f=f, monte_carlo_times=100)  # -174 dBm/Hz with 10 kHz bandwidth
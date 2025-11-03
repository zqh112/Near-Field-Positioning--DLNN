import torch
import numpy as np
import copy

def sense_one_round(model, H: np.ndarray, A: np.ndarray, transmit_power: float, noise_power: float):
    '''
    description: sense via a snapshot for the proposed BiCNN method.
    param {*} model: pretrained NN
    param {np.ndarray} H: round-trip channel matrix
    param {np.ndarray} A: wavenumber-domain transformation matrix
    param {float} transmit_power
    param {float} noise_power
    return {np.ndarray} preds: the estimated position of the target.
    '''
    M = H.shape[0]
    # To obtain the Rx observation:
    # 1. transmit two DL pilots
    noise = np.sqrt(0.5 * noise_power) * np.random.normal(0, 1, (M, 1)) + 1j * np.sqrt(0.5 * noise_power) * np.random.normal(0, 1, (M, 1))
    beamformer = np.sqrt(transmit_power) * (A @  np.ones((M, 1))) / np.linalg.norm((A @  np.ones((M, 1))).flatten(), ord=2)
    observation = A.T.conjugate() @ (H @ beamformer + noise)
    # 2. normalize the received pilots:
    observation = np.abs(observation)
    # 3. feed the normalized pilots to pre-trained NN:
    observation = torch.from_numpy(observation).float().flatten()
    observation = (observation - torch.min(observation)) / (torch.max(observation) - torch.min(observation))
    observation = torch.where(observation>0.5, torch.tensor(1.0), torch.tensor(0.0))
    observation_prime = torch.flip(copy.deepcopy(observation), [-1])
    observation = torch.concat((observation.view(1, 1, -1), observation_prime.view(1, 1, -1)), dim=1)
    model = model.cpu()
    preds = model(observation)

    return preds.detach().cpu().numpy()

if __name__ == '__main__':
    main()
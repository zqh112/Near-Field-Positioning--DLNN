import os
import copy
import torch
from torch.utils.data import Dataset

class EchoSigDataset(Dataset):
    def __init__(self, bidirect:bool=True, dataset_path: str='./samples/CNN'):
        '''
        description: customized dataset for training neural networks
        param {*} self
        param {bool} bidirect: whether or not stack the original observation and the reversed observation on the channel dimension
        param {str} dataset_path: where the dataset is saved.
        return {*}
        '''
        self.label = torch.load(os.path.join(dataset_path, 'label.pth')).float()
        observation_origi_arr = torch.load(os.path.join(dataset_path, 'observation_origi_arr.pth'))
        observation_prime_arr = torch.flip(copy.deepcopy(observation_origi_arr), [1])
        if bidirect:
            # Here the MinMaxscalar is utilized:
            observation_origi_arr = (observation_origi_arr - torch.min(observation_origi_arr, dim=-1, keepdim=True)[0]) / (torch.max(observation_origi_arr, dim=-1, keepdim=True)[0] - torch.min(observation_origi_arr, dim=-1, keepdim=True)[0])
            observation_prime_arr = (observation_prime_arr - torch.min(observation_prime_arr, dim=-1, keepdim=True)[0]) / (torch.max(observation_prime_arr, dim=-1, keepdim=True)[0] - torch.min(observation_prime_arr, dim=-1, keepdim=True)[0])
            observation_origi_arr = torch.where(observation_origi_arr>0.5, torch.tensor(1.0), torch.tensor(0.0))
            observation_prime_arr = torch.where(observation_prime_arr>0.5, torch.tensor(1.0), torch.tensor(0.0))
            observation_origi_arr = observation_origi_arr.unsqueeze(1)
            observation_prime_arr = observation_prime_arr.unsqueeze(1)
            self.observation = torch.concat((observation_origi_arr, observation_prime_arr), dim=1).float()

        else:
            # Original data points:
            observation_origi_arr = (observation_origi_arr - torch.min(observation_origi_arr, dim=-1, keepdim=True)[0]) / (torch.max(observation_origi_arr, dim=-1, keepdim=True)[0] - torch.min(observation_origi_arr, dim=-1, keepdim=True)[0])
            observation_origi_arr = torch.where(observation_origi_arr>0.5, torch.tensor(1.0), torch.tensor(0.0))
            observation_origi_arr = observation_origi_arr.unsqueeze(1)
            self.observation = observation_origi_arr.float()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.observation[idx], self.label[idx]

if __name__ == '__main__':
    main()
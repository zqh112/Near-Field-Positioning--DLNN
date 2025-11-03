import argparse
import os
import torch
import numpy as np
import random
import warnings
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from Dataset.dataset import EchoSigDataset
from model.DLNN import DLNN
from train import train_model
from test import test_model
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(123)
    M = 511  # num of antennas

    bi_echo_signal_dataset = EchoSigDataset(bidirect=True, dataset_path=f'/data/zqh/code/Near-Field-Sensing/samples/CNN-{M}')
    total_count = len(bi_echo_signal_dataset)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_bi_dataset, valid_bi_dataset, test_bi_dataset = torch.utils.data.random_split(
        bi_echo_signal_dataset, (train_count, valid_count, test_count)
    )
    train_bi_loader = DataLoader(train_bi_dataset, batch_size=128, drop_last=True)
    valid_bi_loader = DataLoader(valid_bi_dataset, batch_size=32, drop_last=True)
    test_bi_loader = DataLoader(test_bi_dataset, batch_size=32, drop_last=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Training using {device}')
    model_list = []
    # model_list.append(BiCNN(out_dim=2, num_of_antenna=M).to(device))
    model_list.append(DLNN().to(device))
    # model_list.append(ConvNeXt().to(device))
    number_learnable_params_dict = dict()
    for model in model_list:
        print(f'Train {model.__class__.__name__}:')
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of trainable params: {num_learnable_params}')
        number_learnable_params_dict[model.__class__.__name__] = num_learnable_params
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        train_model(model=model, model_name=model.__class__.__name__, train_loader=train_bi_loader,
                            valid_loader=valid_bi_loader, optimizer=optimizer,
                            lr_scheduler=scheduler, criterion=criterion,
                            patience=10, max_epcoh=10, overwrite=True, device=device,
                            model_path=f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}',
                            data_path=f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}')
    data_path = os.path.join('/data/zqh/code/Near-Field-Sensing/saved_data/', f'CNN-{M}')
    os.makedirs(data_path, exist_ok=True)
    np.save(os.path.join(data_path, 'number_of_param_dict.npy'), number_learnable_params_dict)
    for model in model_list:
        print(f'Test {model.__class__.__name__}:')
        criterion = nn.HuberLoss()
        test_model(model=model, model_name=model.__class__.__name__, test_loader=test_bi_loader, criterion=criterion,
                            device=device, model_path=f'/data/zqh/code/Near-Field-Sensing/saved_model/CNN-{M}')


if __name__ == '__main__':
    main()
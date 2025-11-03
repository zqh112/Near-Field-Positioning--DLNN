import os
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error

def test_model(model, model_name, test_loader, criterion, device, model_path='./saved_model'):
    checkpoint_path = os.path.join(model_path, model_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    preds_list = []
    label_list = []

    with torch.no_grad():
        for obs, label in tqdm(test_loader, desc="Testing", leave=False):
            obs, label = obs.to(device), label.to(device)
            preds = model(obs)
            loss = criterion(preds, label)
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds_list.append(preds.cpu().numpy())
            label_list.append(label.cpu().numpy())

    labels = np.vstack(label_list)
    preds = np.vstack(preds_list)

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(mean_squared_error(labels, preds))

    print(f"Test Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}")
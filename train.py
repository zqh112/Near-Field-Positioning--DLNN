import os
import torch
from tqdm import tqdm
import numpy as np
import pynvml
import time
import threading
from collections import deque


class PowerMonitor:
    """GPU功率监控器"""
    def __init__(self, device_index=0, sample_interval=0.1):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.power_readings = deque()
        self.timestamps = deque()
        self.monitoring = False
        self.monitor_thread = None
        
        # 初始化NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        
    def _monitor_power(self):
        """后台监控功率的线程函数"""
        while self.monitoring:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
                timestamp = time.time()
                
                self.power_readings.append(power)
                self.timestamps.append(timestamp)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"功率监控出错: {e}")
                break
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.power_readings.clear()
            self.timestamps.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_power)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("开始GPU功率监控...")
    
    def stop_monitoring(self):
        """停止监控"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            print("停止GPU功率监控...")
    
    def get_energy_consumption(self):
        """基于时间积分法计算总能耗"""
        if len(self.power_readings) < 2:
            return 0.0, 0.0

        energy = 0.0
        weighted_power_sum = 0.0
        total_duration = 0.0

        for i in range(1, len(self.power_readings)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            power = self.power_readings[i - 1]
            energy += power * dt
            weighted_power_sum += power * dt
            total_duration += dt

        avg_power = weighted_power_sum / total_duration if total_duration > 0 else 0.0
        return energy, avg_power

    
    def get_detailed_stats(self):
        """获取详细统计信息"""
        if len(self.power_readings) == 0:
            return {}
        
        power_list = list(self.power_readings)
        return {
            'samples': len(power_list),
            'avg_power': np.mean(power_list),
            'max_power': np.max(power_list),
            'min_power': np.min(power_list),
            'std_power': np.std(power_list),
            'total_duration': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) >= 2 else 0
        }


def train_model(model, model_name, train_loader, valid_loader, optimizer, lr_scheduler, criterion, max_epcoh,
                device, patience=7,
                overwrite=False, model_path='/data/zqh/code/Near-Field-Sensing/saved_model', data_path='/data/zqh/code/Near-Field-Sensing/saved_data/train',
                monitor_energy=True):
    
    if overwrite:
        print(f'\t (The model will be re-trained...)')
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        
        # 初始化功率监控器
        power_monitor = None
        if monitor_energy and torch.cuda.is_available():
            try:
                power_monitor = PowerMonitor(device_index=0)
                power_monitor.start_monitoring()
            except Exception as e:
                print(f"功率监控初始化失败: {e}")
                power_monitor = None
        
        start_time = time.time()
        best_val_epoch = -1
        valid_loss_list = []
        train_loss_list = []
        
        model.train()
        
        try:
            for epoch in range(max_epcoh):
                train_loss = 0.0
                count = 0
                
                # 训练阶段
                model.train()
                t = tqdm(train_loader, leave=False)
                for (observation, label) in t:
                    observation = observation.to(device)
                    label = label.to(device)
                    
                    preds = model(observation)
                    loss = criterion(preds, label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.detach().cpu().item()
                    count += 1
                    t.set_description(f'Epoch {epoch}: loss={loss.item():4.2f}')
                
                lr_scheduler.step()
                
                # 验证阶段
                valid_loss = validate_model(model, valid_loader, criterion, device=device)
                
                train_loss_list.append(train_loss / count)
                valid_loss_list.append(valid_loss)
                
                print(f'Epoch {epoch}: training loss: {train_loss_list[epoch]:4.2f}, '
                      f'validation loss: {valid_loss_list[epoch]:4.2f}')
                
                # 保存损失数据
                np.save(os.path.join(data_path, f'{model.__class__.__name__}_train_loss.npy'),
                        np.array(train_loss_list))
                np.save(os.path.join(data_path, f'{model.__class__.__name__}_valid_loss.npy'),
                        np.array(valid_loss_list))
                
                # 保存最佳模型
                if len(valid_loss_list) == 1 or valid_loss < valid_loss_list[best_val_epoch]:
                    print(f'\t (New best performance, saving model at epoch {epoch})')
                    torch.save(model.state_dict(), os.path.join(model_path, model_name))
                    best_val_epoch = epoch
                elif best_val_epoch <= epoch - patience:
                    print(f'\t (Early stop due to no improvements over the last {patience} epochs)')
                    break
        
        finally:
            # 停止功率监控并计算能耗
            end_time = time.time()
            total_duration = end_time - start_time
            
            if power_monitor:
                power_monitor.stop_monitoring()
                energy, avg_power = power_monitor.get_energy_consumption()
                stats = power_monitor.get_detailed_stats()
                
                print(f"\n=== 能耗统计 ===")
                print(f"训练总时长: {total_duration:.2f} 秒")
                print(f"总能量消耗: {energy:.2f} 焦耳 ({energy/3600:.6f} kWh)")
                print(f"平均功率: {avg_power:.2f} W")
                print(f"最大功率: {stats['max_power']:.2f} W")
                print(f"最小功率: {stats['min_power']:.2f} W")
                print(f"功率标准差: {stats['std_power']:.2f} W")
                print(f"采样次数: {stats['samples']}")
                
                # 保存能耗数据
                energy_data = {
                    'total_energy_joules': energy,
                    'total_energy_kwh': energy / 3600,
                    'avg_power_watts': avg_power,
                    'max_power_watts': stats['max_power'],
                    'min_power_watts': stats['min_power'],
                    'power_std_watts': stats['std_power'],
                    'training_duration_seconds': total_duration,
                    'samples': stats['samples']
                }
                np.save(os.path.join(data_path, f'{model.__class__.__name__}_energy_consumption.npy'), energy_data)
            else:
                print(f"\n训练总时长: {total_duration:.2f} 秒")
                print("未监控功率消耗")
    
    else:
        print(f'\t (The model will be loaded...)')
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)
        model.load_state_dict(checkpoint)
    
    return model


def validate_model(model, valid_loader, criterion, device):
    """
    在验证集上验证模型
    """
    model.eval()
    avg_loss = []
    
    with torch.no_grad():
        for observation, label in valid_loader:
            observation = observation.to(device)
            label = label.to(device)
            
            preds = model(observation)
            loss = criterion(preds, label)
            avg_loss.append(loss.cpu().item())
    
    return sum(avg_loss) / len(avg_loss)
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 固定采样点
sample_points = [(158, 76), (158, 156), (238, 116), (20, 156), (20, 76), (98, 116), (98, 36), (238, 36)]

def load_field_tensor(file_path_T, file_path_U, file_path_V):
    T = np.load(file_path_T) / 400.0
    U = (np.load(file_path_U) - 0.0) / 0.05
    V = (np.load(file_path_V) - 0.0) / 0.025
    return torch.from_numpy(np.stack([T, U, V], axis=0)).float()  # [3, H, W]

def extract_sparse_tensor(field, sample_points):
    C, H, W = field.shape
    samples = []
    for x, y in sample_points:
        Tuv = field[:, y, x]
        x_norm = x / W
        y_norm = y / H
        sample = torch.cat([Tuv, torch.tensor([x_norm, y_norm])])  # [5]
        samples.append(sample)
    return torch.stack(samples)  # [8, 5]

class FieldSparseWithMaskDataset(Dataset):
    def __init__(self, base_dir, mask_dir='./'):
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.file_names = sorted(os.listdir(os.path.join(base_dir, 'T')))
        self.mask_cache = {}  # 加载后缓存避免重复读取

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        suffix = file_name[2:]  # 去掉 'T_'

        path_T = os.path.join(self.base_dir, 'T', file_name)
        path_U = os.path.join(self.base_dir, 'U', f'U_{suffix}')
        path_V = os.path.join(self.base_dir, 'V', f'V_{suffix}')
        y_field = load_field_tensor(path_T, path_U, path_V)  # [3, H, W]

        sparse_tensor = extract_sparse_tensor(y_field, sample_points)  # [8, 5]

        # 解析半径信息并加载对应掩码
        radius = int(suffix.split('_')[0])  # e.g. '5' from '5_273.15_...'
        if radius not in self.mask_cache:
            mask_path = os.path.join(self.mask_dir, f'mask_r{radius}.npy')
            mask_np = np.load(mask_path)
            self.mask_cache[radius] = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]

        mask_tensor = self.mask_cache[radius]

        return sparse_tensor, mask_tensor, y_field  # [8,5], [1,H,W], [3,H,W]

# ========== 使用示例 ========== #
if __name__ == '__main__':
    base_dir = './data'
    mask_dir = './data/mask'  # 掩码文件在当前目录
    dataset = FieldSparseWithMaskDataset(base_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for sparse_tensor, mask_tensor, y_field in dataloader:
        print("稀疏点 (sparse_tensor):", sparse_tensor.shape)  # [B, 8, 5]
        print("掩码输入 (mask_tensor):", mask_tensor.shape)     # [B, 1, H, W]
        print("真实全场 (y_field):", y_field.shape)           # [B, 3, H, W]
        break


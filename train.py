import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import UNetWithSparse
from dataset import FieldSparseWithMaskDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import csv
import torch.nn.functional as F

def denormalize_field(tensor, T_max=1.0, U_max=0.05, V_max=0.5):
    """
    将归一化后的 T, U, V 恢复为有量纲值
    输入：
        tensor: [3, H, W]，归一化的输出张量
    输出：
        tensor_real: [3, H, W]，反归一化后的张量
    """
    T_norm, U_norm, V_norm = tensor[0], tensor[1], tensor[2]

    # 反归一化
    T_real = T_norm * T_max
    U_real = U_norm * U_max
    V_real = V_norm * V_max

    return torch.stack([T_real, U_real, V_real], dim=0)

# ====== 自定义 Masked MSE 损失函数 ======
def masked_mse_loss(pred, target, mask):
    """
    pred, target: [B, 3, H, W]
    mask:         [B, 1, H, W]
    """
    mask = mask.expand_as(pred)  # [B, 3, H, W]
    diff = (pred - target) ** 2 * mask

    loss_T = diff[:, 0].sum() / (mask[:, 0].sum() + 1e-8)
    loss_U = diff[:, 1].sum() / (mask[:, 1].sum() + 1e-8)
    loss_V = diff[:, 2].sum() / (mask[:, 2].sum() + 1e-8)

    return loss_T, loss_U, loss_V

# ====== 自定义 物理残差损失 ======
'''def compute_physics_loss(pred, mask, dx=0.005, dy=0.005, alpha=1.43e-7):
    T = pred[:, 0]
    u = pred[:, 1]
    v = pred[:, 2]

    dT_dy, dT_dx = torch.gradient(T, spacing=(dy, dx), dim=(1, 2))
    dv_dy, _ = torch.gradient(v, spacing=(dy, dx), dim=(1, 2))
    _, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2))

    d2T_dy2, _ = torch.gradient(dT_dy, spacing=(dy, dx), dim=(1, 2))
    _, d2T_dx2 = torch.gradient(dT_dx, spacing=(dy, dx), dim=(1, 2))

    R_energy = u * dT_dx + v * dT_dy - alpha * (d2T_dx2 + d2T_dy2)
    R_mass = du_dx + dv_dy

    mask = mask.squeeze(1)
    R_energy = R_energy * mask
    R_mass = R_mass * mask

    loss_energy = (R_energy ** 2).sum() / (mask.sum() + 1e-8)
    loss_mass = (R_mass ** 2).sum() / (mask.sum() + 1e-8)

    return loss_energy, loss_mass'''

def diff_x_center(f, dx):
    f_pad = F.pad(f, (1, 1, 0, 0), mode='reflect')
    return (f_pad[:, :, 2:] - f_pad[:, :, :-2]) / (2 * dx)

def diff_y_center(f, dy):
    f_pad = F.pad(f, (0, 0, 1, 1), mode='reflect')
    return (f_pad[:, 2:, :] - f_pad[:, :-2, :]) / (2 * dy)

def diff2_x_center(f, dx):
    f_pad = F.pad(f, (1, 1, 0, 0), mode='reflect')
    return (f_pad[:, :, 2:] - 2 * f + f_pad[:, :, :-2]) / (dx ** 2)

def diff2_y_center(f, dy):
    f_pad = F.pad(f, (0, 0, 1, 1), mode='reflect')
    return (f_pad[:, 2:, :] - 2 * f + f_pad[:, :-2, :]) / (dy ** 2)

# ====== 重写 compute_physics_loss ======
def compute_physics_loss(pred, mask, dx=0.005, dy=0.005, alpha=1.43e-7):
    """
    pred: [B, 3, H, W]
    mask: [B, 1, H, W]
    """
    T = pred[:, 0]  # [B, H, W]
    u = pred[:, 1]
    v = pred[:, 2]

    dT_dx = diff_x_center(T, dx)
    dT_dy = diff_y_center(T, dy)
    du_dx = diff_x_center(u, dx)
    dv_dy = diff_y_center(v, dy)
    d2T_dx2 = diff2_x_center(T, dx)
    d2T_dy2 = diff2_y_center(T, dy)

    R_energy = u * dT_dx + v * dT_dy - alpha * (d2T_dx2 + d2T_dy2)
    R_mass = du_dx + dv_dy

    mask = mask.squeeze(1)  # [B, H, W]
    R_energy = R_energy * mask
    R_mass = R_mass * mask

    loss_energy = (R_energy ** 2).sum() / (mask.sum() + 1e-8)
    loss_mass = (R_mass ** 2).sum() / (mask.sum() + 1e-8)

    return loss_energy, loss_mass

# ====== 单个 Epoch 的训练函数 ======
def train_one_epoch(model, dataloader, optimizer, device,
                    weight_T=1.0, weight_U=1.0, weight_V=1.0,
                    weight_phys_energy=1.0, weight_phys_mass=1.0,
                    lambda_data=1.0, lambda_phys=1.0):
    model.train()
    total_loss = 0
    total_loss_T = 0
    total_loss_U = 0
    total_loss_V = 0
    total_phys_energy = 0
    total_phys_mass = 0

    for sparse_tensor, mask_tensor, y_field in dataloader:
        sparse_tensor = sparse_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        y_field = y_field.to(device)

        optimizer.zero_grad()
        pred = model(sparse_tensor, mask_tensor)

        loss_T, loss_U, loss_V = masked_mse_loss(pred, y_field, mask_tensor)
        data_loss = weight_T * loss_T + weight_U * loss_U + weight_V * loss_V

        pred_denorm = torch.stack([
            denormalize_field(pred[i]) for i in range(pred.shape[0])
        ], dim=0)  # [B, 3, H, W]
        phys_loss_energy, phys_loss_mass = compute_physics_loss(pred_denorm, mask_tensor)
        total_phys_loss = weight_phys_energy * phys_loss_energy + weight_phys_mass * phys_loss_mass

        loss = lambda_data * data_loss + lambda_phys * total_phys_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_T += loss_T.item()
        total_loss_U += loss_U.item()
        total_loss_V += loss_V.item()
        total_phys_energy += phys_loss_energy.item()
        total_phys_mass += phys_loss_mass.item()

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'loss_T': total_loss_T / n_batches,
        'loss_U': total_loss_U / n_batches,
        'loss_V': total_loss_V / n_batches,
        'loss_energy': total_phys_energy / n_batches,
        'loss_mass': total_phys_mass / n_batches,
        'data_loss': data_loss/ n_batches,
        'phys_loss': total_phys_loss.item()

    }

# ====== 主程序入口 ======
def main():
    # 假设你在每个 epoch 后把 total loss 存入这个列表
    all_total_losses = []
    all_data_losses = []
    all_T_losses = []
    all_U_losses = []
    all_V_losses = []
    all_phys_losses = []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FieldSparseWithMaskDataset(base_dir='./data', mask_dir='./data/mask')
    # 设置划分比例（你可以自定义为可调变量）
    train_ratio = 0.5
    # 样本总数
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size-int(train_ratio * total_size)
    print("样本数目",train_size)
    # 使用 random_split 划分数据集
    dataset,_ = random_split(dataset, [train_size,test_size])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


    model = UNetWithSparse().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    # 权重设置
    weight_T = 10.0
    weight_U = 1.0
    weight_V = 1.0
    weight_phys_energy = 0.1
    weight_phys_mass = 0.0
    lambda_data = 1.0
    lambda_phys = 1.0

    best_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in range(1, 501):

        losses = train_one_epoch(model, dataloader, optimizer, device,
                                 weight_T, weight_U, weight_V,
                                 weight_phys_energy, weight_phys_mass,
                                 lambda_data, lambda_phys)
        #scheduler.step()

        print(f"[Epoch {epoch:03d}] "
              f"Total Loss: {losses['total_loss']:.6f} | "
              f"T Loss: {losses['loss_T']:.6f} | "
              f"U Loss: {losses['loss_U']:.6f} | "
              f"V Loss: {losses['loss_V']:.6f} | "
              f"Energy Phys Loss: {losses['loss_energy']:.6f} | "
              f"Mass Phys Loss: {losses['loss_mass']:.6f}")

        all_total_losses.append(losses['total_loss'])
        all_data_losses.append(losses['loss_T'] + losses['loss_U'] + losses['loss_V'])  # 数据损失
        all_T_losses.append(losses['loss_T'])  # 数据损失
        all_U_losses.append(losses['loss_U'])  # 数据损失
        all_V_losses.append(losses['loss_V'])  # 数据损失
        all_phys_losses.append(losses['loss_energy'])  # 物理残差

        # ✅ 保存最优模型
        if losses['total_loss'] < best_loss:
            best_loss = losses['total_loss']
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 保存新的最佳模型 (Epoch {epoch}) Loss = {best_loss:.6f}")

    print("✅ 训练完成，最佳模型已保存为 best_model.pth")

    # ✅ 保存最后一个 epoch 的模型（非最优）
    torch.save(model.state_dict(), "last_model.pth")
    print("📦 已保存最后一个模型权重为 last_model.pth")

    # 📌 设置字体和风格
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12

    # 📌 主图显示 epoch > 50
    if len(all_total_losses) > 50:
        fig, ax = plt.subplots(figsize=(8, 4))

        epochs = np.arange(1, len(all_total_losses) + 1)
        ax.plot(epochs[50:], all_total_losses[50:], label='Total Loss', color='blue', linewidth=2)
        ax.plot(epochs[50:], all_data_losses[50:], label='Data Loss', color='green', linestyle='--', linewidth=1.5)
        ax.plot(epochs[50:], all_phys_losses[50:], label='Physics Loss', color='red', linestyle='--', linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.set_title("Training Loss Curve (Epoch 1–50 Omitted)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 🔍 插入小图：前50 epoch Total Loss
        ax_inset = ax.inset_axes([0.6, 0.55, 0.35, 0.35])
        ax_inset.plot(epochs[:50], all_total_losses[:50], color='gray', linestyle='--', linewidth=1.5)
        ax_inset.set_title("Epoch 1–50", fontsize=10)
        ax_inset.tick_params(labelsize=8)
        ax_inset.grid(True, alpha=0.2)

        plt.tight_layout()
        os.makedirs("vis_result", exist_ok=True)
        plt.savefig("vis_result/loss_components.png", dpi=300)
        plt.close()
        print("📉 保存 loss_components.png 成功")
    else:
        print("⚠️ Epoch < 50，未绘图")

    # 创建保存目录
    os.makedirs("vis_result", exist_ok=True)
    csv_path = "vis_result/loss_curve.csv"

    # 写入 CSV 文件
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Data Loss", "T Loss", "U Loss", "V Loss", "Physics Loss"])

        for i, (data_loss, t_loss, u_loss, v_loss, phys_loss) in enumerate(
                zip(all_data_losses, all_T_losses, all_U_losses, all_V_losses, all_phys_losses)):
            writer.writerow([i + 1, data_loss, t_loss, u_loss, v_loss, phys_loss])

    print(f"✅ 已保存训练 Loss 曲线数据到：{csv_path}")

if __name__ == '__main__':
    main()

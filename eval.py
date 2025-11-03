import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import UNetWithSparse
from dataset import FieldSparseWithMaskDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter  # ✅ 用于设置colorbar数字格式
def compute_mape(pred, true):
    mask = (true != 0)
    return (np.abs((pred[mask] - true[mask]) / true[mask])).mean() * 100

# 🔵 加入指标计算函数
#梯度均方（顺滑性）
def compute_grad_mse(field):
    grad_x = torch.gradient(field, dim=1)[0]
    grad_y = torch.gradient(field, dim=0)[0]
    grad_mse = (grad_x ** 2 + grad_y ** 2).mean()
    return grad_mse.item()

#拉普拉斯均方（二阶顺滑）
def compute_laplace_mse(field):
    grad_x = torch.gradient(field, dim=1)[0]
    grad_y = torch.gradient(field, dim=0)[0]
    laplace_x = torch.gradient(grad_x, dim=1)[0]
    laplace_y = torch.gradient(grad_y, dim=0)[0]
    laplace = laplace_x + laplace_y
    laplace_mse = (laplace ** 2).mean()
    return laplace_mse.item()

#稳态PDE残差均方
def compute_pde_residual_mse_steady(T, U, V, alpha=1.0):
    T_x = torch.gradient(T, dim=1)[0]
    T_y = torch.gradient(T, dim=0)[0]
    T_xx = torch.gradient(T_x, dim=1)[0]
    T_yy = torch.gradient(T_y, dim=0)[0]
    convection = U * T_x + V * T_y
    diffusion = alpha * (T_xx + T_yy)
    residual = convection - diffusion
    residual_mse = (residual ** 2).mean()
    return residual_mse.item()

#稳态PDE残差标准差
def compute_pde_residual_std_steady(T, U, V, alpha=1.0):
    T_x = torch.gradient(T, dim=1)[0]
    T_y = torch.gradient(T, dim=0)[0]
    T_xx = torch.gradient(T_x, dim=1)[0]
    T_yy = torch.gradient(T_y, dim=0)[0]
    convection = U * T_x + V * T_y
    diffusion = alpha * (T_xx + T_yy)
    residual = convection - diffusion
    residual_std = torch.std(residual)
    return residual_std.item()

#高频噪声能量占比
def compute_high_frequency_energy_ratio(field, freq_threshold_ratio=0.5):
    fft_field = torch.fft.fft2(field)
    fft_magnitude = torch.abs(fft_field) ** 2
    H, W = field.shape
    freq_H = torch.fft.fftfreq(H)
    freq_W = torch.fft.fftfreq(W)
    freq_H_grid, freq_W_grid = torch.meshgrid(freq_H, freq_W, indexing="ij")
    freq_magnitude = torch.sqrt(freq_H_grid ** 2 + freq_W_grid ** 2)

    freq_cutoff = freq_threshold_ratio * freq_magnitude.max()
    high_freq_energy = fft_magnitude[freq_magnitude >= freq_cutoff].sum()
    total_energy = fft_magnitude.sum()
    high_freq_ratio = (high_freq_energy / total_energy).item()
    return high_freq_ratio

#局部变化率（局部震荡程度）
def compute_local_variation(field, kernel_size=5):
    pad = kernel_size // 2
    field_padded = torch.nn.functional.pad(field.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
    unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size))
    patches = unfold(field_padded).squeeze(0).T
    patch_variances = torch.var(patches, dim=1)
    local_variation = patch_variances.mean()
    return local_variation.item()

#质量守恒残差
def compute_continuity_residual(U, V):
    U_x = torch.gradient(U, dim=1)[0]
    V_y = torch.gradient(V, dim=0)[0]
    divergence = U_x + V_y
    continuity_mse = (divergence ** 2).mean()
    return continuity_mse.item()

#边界条件误差
def compute_boundary_condition_error(field, true_boundary_value, boundary="left"):
    if boundary == "left":
        boundary_pred = field[:, 0]
    elif boundary == "right":
        boundary_pred = field[:, -1]
    elif boundary == "top":
        boundary_pred = field[0, :]
    elif boundary == "bottom":
        boundary_pred = field[-1, :]
    else:
        raise ValueError("Boundary must be 'left', 'right', 'top', or 'bottom'")
    error = ((boundary_pred - true_boundary_value) ** 2).mean()
    return error.item()


def plot_prediction_vs_groundtruth_main(pred, truth, mask, save_path=None):
    # 📌 设置全局字体
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12  # 标准正文图字号
    rcParams['axes.titlesize'] = 14  # 标题字号
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10

    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()
    H, W = mask.shape

    masked_pred = np.where(mask[None, :, :] == 1, pred, np.nan)
    masked_truth = np.where(mask[None, :, :] == 1, truth, np.nan)

    names = ['T', 'u', 'v']
    cmaps = ['coolwarm', 'jet', 'jet']
    extent = [0, 257, 0, 193]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    for i in range(3):
        vmin = np.nanmin([masked_pred[i], masked_truth[i]])
        vmax = np.nanmax([masked_pred[i], masked_truth[i]])

        im0 = axes[0, i].imshow(masked_pred[i], cmap=cmaps[i], origin='lower',
                                extent=extent, vmin=vmin, vmax=vmax)
        im1 = axes[1, i].imshow(masked_truth[i], cmap=cmaps[i], origin='lower',
                                extent=extent, vmin=vmin, vmax=vmax)

        axes[0, i].set_title(f'Predicted {names[i]}', fontweight='bold')
        axes[1, i].set_title(f'True {names[i]}', fontweight='bold')

        for ax in axes[:, i]:
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect('equal')
            ax.axis('off')

        # 添加 colorbar 且保持比例紧凑
        cbar0 = fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
        cbar1 = fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
        cbar0.ax.tick_params(labelsize=10)
        cbar1.ax.tick_params(labelsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved figure at {save_path}")
    plt.show()

def plot_streamlines(pred, truth, mask, save_path='vis_result/streamlines_masked.png'):
    """
    绘制只在掩码为1区域的流线图，掩码为0（柱体）区域不显示流线。
    """
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()  # [H, W]
    H, W = mask.shape

    # 获取预测与真实的速度场
    u_pred, v_pred = pred[1], pred[2]
    u_truth, v_truth = truth[1], truth[2]

    # 计算速度大小
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    speed_truth = np.sqrt(u_truth**2 + v_truth**2)

    # 使用掩码屏蔽柱体区域（掩码为0的区域）
    u_pred_masked = np.ma.masked_where(mask == 0, u_pred)
    v_pred_masked = np.ma.masked_where(mask == 0, v_pred)
    speed_pred_masked = np.ma.masked_where(mask == 0, speed_pred)

    u_truth_masked = np.ma.masked_where(mask == 0, u_truth)
    v_truth_masked = np.ma.masked_where(mask == 0, v_truth)
    speed_truth_masked = np.ma.masked_where(mask == 0, speed_truth)

    # 构建坐标网格
    X, Y = np.meshgrid(np.linspace(0, W, W), np.linspace(0, H, H))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 预测流线图
    axes[0].streamplot(X, Y, u_pred_masked, v_pred_masked, color=speed_pred_masked,
                       cmap='viridis', density=1.2, linewidth=1)
    axes[0].set_title('Predicted Streamlines')
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(0, H)
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # 真实流线图
    axes[1].streamplot(X, Y, u_truth_masked, v_truth_masked, color=speed_truth_masked,
                       cmap='viridis', density=1.2, linewidth=1)
    axes[1].set_title('True Streamlines')
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(0, H)
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved streamline figure (with masked pillars): {save_path}")
    plt.show()

def denormalize_field(tensor):
    # tensor 形状 [3, H, W]
    T = tensor[0] * 400.0
    U = tensor[1] * 0.05
    V = tensor[2] * 0.025
    return torch.stack([T, U, V], dim=0)


# 🔵 主程序，增加指标计算
def run_prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetWithSparse()
    model.load_state_dict(torch.load('last_model.pth', map_location=device))
    model.to(device)
    model.eval()

    pred_dataset = FieldSparseWithMaskDataset(base_dir='./pred', mask_dir='./data/mask')
    os.makedirs("vis_result", exist_ok=True)

    # 收集所有样本指标
    all_metrics = []

    #indices = np.random.choice(len(pred_dataset), size=16, replace=False)
    indices = np.arange(16)

    for local_idx, global_idx in enumerate(indices):
        sparse_tensor, mask_tensor, y_field = pred_dataset[global_idx]
        sparse_tensor = sparse_tensor.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)
        y_field = y_field.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sparse_tensor, mask_tensor)

        pred = output[0]
        truth = y_field[0]
        mask = mask_tensor[0]

        # 反归一化
        pred = denormalize_field(pred)
        truth = denormalize_field(truth)

        save_path = f"./vis_result/newsample_{local_idx:02d}.png"
        plot_prediction_vs_groundtruth_main(pred, truth, mask, save_path=save_path)
        #plot_streamlines(pred, truth, mask, save_path=f"./vis_result/sample_{local_idx:02d}_streamline.png")

        # 只对掩码内有效区域进行指标计算
        T_pred, U_pred, V_pred = pred[0] * mask[0], pred[1] * mask[0], pred[2] * mask[0]
        T_truth, U_truth, V_truth = truth[0] * mask[0], truth[1] * mask[0], truth[2] * mask[0]

        # 在每次循环中加入
        mse = mean_squared_error(T_truth.cpu().numpy().flatten(), T_pred.cpu().numpy().flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(T_truth.cpu().numpy().flatten(), T_pred.cpu().numpy().flatten())
        mape = compute_mape(T_pred.cpu().numpy(), T_truth.cpu().numpy())

        metrics = {
            'T-MSE': mse,
            'T-RMSE': rmse,
            'T-MAE': mae,
            'T-MAPE': mape,
            'T-GradMSE': compute_grad_mse(T_pred),
            'T-LapMSE': compute_laplace_mse(T_pred),
            'U-GradMSE': compute_grad_mse(U_pred),
            'V-GradMSE': compute_grad_mse(V_pred),
            'PDE-Residual-MSE': compute_pde_residual_mse_steady(T_pred, U_pred, V_pred),
            'PDE-Residual-STD': compute_pde_residual_std_steady(T_pred, U_pred, V_pred),
            'HighFreqEnergyRatio(T)': compute_high_frequency_energy_ratio(T_pred),
            'LocalVariation(T)': compute_local_variation(T_pred),
            'ContinuityResidual(UV)': compute_continuity_residual(U_pred, V_pred),
            'BoundaryError(T-left)': compute_boundary_condition_error(T_pred, true_boundary_value=1.0, boundary='left'), # ⚡这里改成你的真实入口温度
        }

        all_metrics.append(metrics)

    # 输出指标
    for idx, metric in enumerate(all_metrics):
        print(f"Sample {idx}:")
        for k, v in metric.items():
            print(f"  {k}: {v:.6f}")
        print()

    # 🔵 统计每个指标的平均值
    keys = all_metrics[0].keys()
    avg_metrics = {k: 0.0 for k in keys}

    for metric in all_metrics:
        for k, v in metric.items():
            avg_metrics[k] += v

    for k in avg_metrics:
        avg_metrics[k] /= len(all_metrics)

    print("🔵 Overall Average Metrics (over 10 samples):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.6f}")

    plot_keys = ['T-MSE', 'T-RMSE', 'T-MAE', 'T-MAPE']
    for key in plot_keys:
        values = [m[key] for m in all_metrics]
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(values)), values)
        plt.xlabel('Sample Index')
        plt.ylabel(key)
        plt.title(f'{key} Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'vis_result/distribution_{key}.png', dpi=300)
        plt.show()



if __name__ == '__main__':
    run_prediction()

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import FieldSparseWithMaskDataset
from torch.utils.data import DataLoader

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetEnhancedWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=16):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)

        self.center = self.conv_block(base_filters * 4, base_filters * 8)

        self.att3 = AttentionGate(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.att2 = AttentionGate(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)

        self.dec3 = self.conv_block(base_filters * 8, base_filters * 4)
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)
        self.dec1 = self.conv_block(base_filters * 2, base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        center = self.center(F.max_pool2d(e3, 2))

        d3 = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=False)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # 这里e1小，不加注意力也行，加也可以

        return self.final(d1)


class UNetEnhanced(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16, norm='group', activation='relu'):
        super().__init__()
        self.norm = norm
        self.activation = activation

        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)

        # ⚠️ 新增 center 模块（更强的表达）
        self.center = self.conv_block(base_filters * 4, base_filters * 8)

        self.dec3 = self.conv_block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self.conv_block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self.conv_block(base_filters * 2 + base_filters, base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            self.get_norm(out_channels),
            self.get_act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            self.get_norm(out_channels),
            self.get_act()
        ]
        return nn.Sequential(*layers)

    def get_norm(self, num_channels):
        if self.norm == 'group':
            return nn.GroupNorm(8, num_channels)
        elif self.norm == 'batch':
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.Identity()

    def get_act(self):
        if self.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        center = self.center(F.max_pool2d(e3, 2))  # ⬅️ 更深的瓶颈

        d3 = self.dec3(torch.cat([F.interpolate(center, size=e3.shape[-2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], dim=1))

        return self.final(d1)  # [B, 3, H, W]

class PointEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU()
        )

    def forward(self, x):  # x: [B, N, 5]
        return self.encoder(x)  # [B, N, out_dim]


class SparseMLPMap(nn.Module):
    def __init__(self, point_dim=5, point_feature_dim=128, fusion_hidden=256, H=193, W=257, C=3):
        super().__init__()
        self.H, self.W, self.C = H, W, C

        # 编码每个采样点
        self.point_encoder = PointEncoder(in_dim=point_dim, hidden_dim=64, out_dim=point_feature_dim)

        # 汇聚所有点的编码，输出整图
        self.fusion = nn.Sequential(
            nn.Linear(point_feature_dim * 8, fusion_hidden),  # 假设有8个采样点
            nn.GELU(),
            nn.Linear(fusion_hidden, C * H * W)
        )

    def forward(self, sparse_tensor):
        """
        sparse_tensor: [B, 8, 5]
        """
        B = sparse_tensor.shape[0]
        encoded_points = self.point_encoder(sparse_tensor)  # [B, 8, point_feature_dim]
        x = encoded_points.view(B, -1)                      # [B, 8 * dim]
        x = self.fusion(x)                                  # [B, C×H×W]
        return x.view(B, self.C, self.H, self.W)            # [B, C, H, W]


# ========= 精简后的 Sparse MLP 编码器 =========
class SparseMLPMapLite(nn.Module):
    def __init__(self, point_dim=5, point_feature_dim=32, fusion_hidden=64, out_H=48, out_W=64, out_C=3):
        super().__init__()
        self.out_H, self.out_W, self.out_C = out_H, out_W, out_C
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(),
            nn.Linear(64, point_feature_dim),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(point_feature_dim * 8, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, out_C * out_H * out_W))
        

    def forward(self, sparse_tensor):  # [B, 8, 5]
        B = sparse_tensor.shape[0]
        x = self.point_encoder(sparse_tensor)      # [B, 8, feature_dim]
        x = x.view(B, -1)                           # [B, 8*feature_dim]
        x = self.fusion(x)                          # [B, C × H' × W']
        x = x.view(B, self.out_C, self.out_H, self.out_W)  # [B, 3, H', W']
        x = F.interpolate(x, size=(193, 257), mode='bilinear', align_corners=False)  # 上采样到全尺寸
        return x

class UNetWithSparse(nn.Module):
    def __init__(self, H=193, W=257):
        super().__init__()
        self.mlp_encoder = SparseMLPMapLite(out_H=48, out_W=64, out_C=3)
        self.unet = UNetEnhanced(in_channels=4, out_channels=3)

    def forward(self, sparse_tensor, mask_tensor):  # sparse_tensor: [B, 8, 5]
        x_init = self.mlp_encoder(sparse_tensor)  # [B, 3, H, W]
        x = torch.cat([x_init, mask_tensor], dim=1)  # 拼接掩码 → [B, 4, H, W]
        out = self.unet(x)                   # [B, 3, H, W]
        return out

if __name__ == '__main__':
    # 模拟输入数据并运行模型
    dataset = FieldSparseWithMaskDataset(base_dir='./data', mask_dir='./data/mask')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNetWithSparse()

    for sparse_tensor, mask_tensor, y_field in dataloader:
        output = model(sparse_tensor, mask_tensor)  # 输出形状 [32, 3, 193, 257]
        print("预测输出:", output.shape)
        break

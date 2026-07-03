import torch
import numpy as np
from inr2vec.inr_step1.model import INR, RFFPE
from inr2vec.inr_step1.train import train_inr

def create_random_circle_sdf(resolution=128):
    """随机生成一个圆（半径在 0.3 到 0.6 之间随机）"""
    radius = np.random.uniform(0.3, 0.6)
    xs = torch.linspace(-1, 1, resolution)
    ys = torch.linspace(-1, 1, resolution)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
    distances = torch.linalg.norm(coords, dim=-1) - radius
    return coords, distances.unsqueeze(-1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Max 建议测试的样本数
    num_samples = 64 
    # 我们测试 3 个不同的随机种子来看看波动有多大
    seeds_to_test = [42, 123, 2026] 
    
    print(f"🧪 开始对 RFFPE 进行种子消融实验 (每个种子跑 {num_samples} 个样本)...")

    for seed in seeds_to_test:
        psnr_list = []
        # 固定当前的 RFFPE 随机生成器种子
        print(f"\n🌱 正在测试 Seed: {seed} ...")
        
        # 为了速度，我们把分辨率降到 128，epochs 设为 300 快速看结果
        for i in range(num_samples):
            coords, targets = create_random_circle_sdf(resolution=128)
            coords, targets = coords.to(device), targets.to(device)
            
            model = INR(
                pe=RFFPE, # 强行使用 RFFPE
                seed=seed, # 注入当前测试的种子
                hidden_dim=64,
                hidden_layers=3,
                out_dim=1,
                num_frequencies=128,
                num_bands=8,
                sigma=10.0
            ).to(device)
            
            psnr = train_inr(model=model, coords=coords, target=targets, epochs=300, lr=1e-3, patience=20)
            psnr_list.append(psnr)
            
        avg_psnr = np.mean(psnr_list)
        print(f"✅ Seed {seed} 完成！{num_samples} 个样本的平均 PSNR = {avg_psnr:.2f} dB")
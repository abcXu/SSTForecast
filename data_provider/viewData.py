import matplotlib.pyplot as plt
import numpy as np
import os

# 加载数据
sst_path = 'D:/GraduationThesis/codes/preData/part_data/sst_Y.npy'
ssh_path = 'D:/GraduationThesis/codes/preData/part_data/ssh_X.npy'
mask_path = 'D:/GraduationThesis/codes/preData/maskLand=0.npy'

sst_one_sample = np.load(sst_path)[10]  # [10, 1, 64, 64]
ssh_one_sample = np.load(ssh_path)[0]  # [10, 1, 64, 64]
mask = np.load(mask_path)  # [64, 64]

# 帧索引（第1, 2, 10帧）
frames_to_plot = [0, 1, 9]

# 输出文件夹
output_dir = "./output_vis"
os.makedirs(output_dir, exist_ok=True)


# 通用绘图函数
def plot_single_frame(data, mask, variable_name, frame_idx, cmap='jet'):
    frame = data[frame_idx, 0]  # [64, 64]
    masked_frame = np.ma.masked_where(mask == 0, frame)

    plt.figure(figsize=(4, 4))
    plt.imshow(masked_frame, cmap=cmap, origin='lower', alpha=0.9)
    plt.axis('off')
    plt.tight_layout()

    filename = f"{variable_name}_frame{frame_idx + 1}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"Saved: {save_path}")


# 绘制并保存每帧
for idx in frames_to_plot:
    plot_single_frame(sst_one_sample, mask, "SST", idx)
    plot_single_frame(ssh_one_sample, mask, "SSH", idx)

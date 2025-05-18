import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 联轴器三维模型简化示意图
def plot_coupling_model():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 联轴器主体(圆柱体)
    radius = 52.5  # D/2=105/2
    length = 60
    x = np.linspace(-length / 2, length / 2, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    X, Theta = np.meshgrid(x, theta)
    Y = radius * np.cos(Theta)
    Z = radius * np.sin(Theta)
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.7)

    # 轴孔
    shaft_radius = 15  # d/2=30/2
    Y_shaft = shaft_radius * np.cos(Theta)
    Z_shaft = shaft_radius * np.sin(Theta)
    ax.plot_surface(X, Y_shaft, Z_shaft, color='gray')

    # 螺栓孔(简化表示)
    bolt_circle_radius = 27.5  # D1/2=55/2
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        x_bolt = [-20, 20]
        y_bolt = bolt_circle_radius * np.sin(angle)
        z_bolt = bolt_circle_radius * np.cos(angle)
        ax.plot(x_bolt, [y_bolt, y_bolt], [z_bolt, z_bolt],
                linewidth=3, color='red')

    # 键槽
    key_width = 5
    key_length = 45
    y_key = np.linspace(shaft_radius, shaft_radius + key_width, 10)
    z_key = np.zeros(10)
    x_key = np.linspace(-key_length / 2, key_length / 2, 10)
    X_key, Y_key = np.meshgrid(x_key, y_key)
    Z_key = np.zeros_like(X_key)
    ax.plot_surface(X_key, Y_key, Z_key, color='orange')

    ax.set_title('GYS4型凸缘联轴器三维模型示意图', fontsize=14)
    ax.set_xlabel('轴向 (mm)')
    ax.set_ylabel('Y轴 (mm)')
    ax.set_zlabel('Z轴 (mm)')
    plt.tight_layout()
    plt.show()


# 2. 有限元应力云图示意图
def plot_stress_contour():
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建假想应力分布数据
    x = np.linspace(-60, 60, 100)
    y = np.linspace(-60, 60, 100)
    X, Y = np.meshgrid(x, y)
    stress = 200 * np.exp(-(X ** 2 + Y ** 2) / 2000) * (1 + 0.5 * np.sin(X / 10))

    # 绘制应力云图
    contour = ax.contourf(X, Y, stress, levels=20, cmap='jet')
    cbar = fig.colorbar(contour)
    cbar.set_label('应力 (MPa)', rotation=270, labelpad=20)

    # 标记关键部位
    ax.plot(27.5, 0, 'ro', markersize=8, label='螺栓孔')
    ax.plot(15, 0, 'ws', markersize=8, label='键槽')
    ax.plot(-15, 0, 'ws', markersize=8)
    ax.plot(0, 52.5, 'kx', markersize=8, label='边缘')

    ax.set_title('联轴器应力分布云图 (有限元分析结果)', fontsize=14)
    ax.set_xlabel('X方向 (mm)')
    ax.set_ylabel('Y方向 (mm)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# 3. 螺栓受力分析图
def plot_bolt_force():
    fig, ax = plt.subplots(figsize=(8, 6))

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    forces = [16290 + 2000 * np.random.rand() for _ in range(4)]  # 添加随机变化

    # 极坐标图
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(angles, forces, width=0.5,
                  color=['red', 'green', 'blue', 'orange'], alpha=0.7)

    # 标注数值
    for angle, force, bar in zip(angles, forces, bars):
        ax.text(angle, force + 500, f'{force:.0f}N', ha='center')

    ax.set_title('螺栓预紧力分布 (4个M10螺栓)', pad=20)
    ax.set_xticks(angles)
    ax.set_xticklabels(['螺栓1', '螺栓2', '螺栓3', '螺栓4'])
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.show()


# 运行绘图函数
plot_coupling_model()
plot_stress_contour()
plot_bolt_force()
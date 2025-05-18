import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# 设置字体以及负号正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读数据 & 掩码
eff_matrix = np.load("npys/new_R2_0.88&RMSE_0.7.matrix.npy")   # [64×64] 有效时长 0–10
mask = np.load("D:/GraduationThesis/codes/preData/maskLand=0.npy")              # 0=陆地,1=海洋

# 构造经纬度网格
lon = np.linspace(107.125, 122.875, 64)
lat = np.linspace(8.125,  23.875, 64)
lon2d, lat2d = np.meshgrid(lon, lat)

# 掩蔽陆地  land=0
eff_masked = ma.masked_array(eff_matrix, mask==0)

# 画底图 + 热力图
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.set_extent([107,123,8,24])
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False   # 去掉顶部标签
gl.right_labels = False  # 去掉右侧标签

# 颜色层级
levels = np.linspace(0,10,11)
cf = ax.contourf(lon2d, lat2d, eff_masked, levels=levels,cmap="coolwarm", vmin=0, vmax=10, extend='both')
# 颜色条 & 标题
cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.7,
                    ticks=np.arange(0,11,1))
cbar.set_label('有效预测时长（天）', fontsize=12)
cbar.ax.tick_params(labelsize=10)
ax.set_title('预测有效时长热力图', fontsize=14)

plt.tight_layout()
plt.savefig('pictures/有效预测时长热力图-2.png', dpi=300, bbox_inches='tight', format='png')
plt.show()






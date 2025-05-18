import numpy as np
from sklearn.metrics import r2_score

# pred_path = "../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy"
# true_path = "../模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy"

pred_path = "D:/Datasets/result/20250415/reversed/reverse_pred_sst_SSTPredictor_mlp_SCS.npy"
true_path = "D:/Datasets/result/20250415/reversed/reverse_true_sst_SSTPredictor_mlp_SCS.npy"

# 加载真实数据和预测数据
true_data = np.load(true_path)
pred_data = np.load(pred_path)
N, T, H, W = true_data.shape


# def r2_score_per_position(pred_data, true_data):
#     # Step 1: Calculate the mean of the true values at each position
#     true_mean = np.mean(true_data, axis=0)
#
#     # Step 2: Calculate SS_tot (total sum of squares)
#     ss_tot = np.sum((true_data - true_mean) ** 2, axis=0)
#
#     # Step 3: Calculate SS_res (residual sum of squares)
#     ss_res = np.sum((true_data - pred_data) ** 2, axis=0)
#
#     # Step 4: Calculate R^2
#     r2 = 1 - (ss_res / ss_tot)
#
#     return r2
def r2_and_rmse_per_position(pred_data, true_data):
    # Calculate the mean of the true values at each position
    true_mean = np.mean(true_data, axis=0)

    # Calculate SS_tot (total sum of squares)
    ss_tot = np.sum((true_data - true_mean) ** 2, axis=0)

    # Calculate SS_res (residual sum of squares)
    ss_res = np.sum((true_data - pred_data) ** 2, axis=0)

    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)

    # Calculate RMSE
    mse = np.mean((true_data - pred_data) ** 2, axis=0)
    rmse = np.sqrt(mse)

    return r2, rmse
r2_matrix ,rmse_matrix = r2_and_rmse_per_position(pred_data, true_data)
# 在N的维度上取均值
# true_data = np.mean(true_data, axis=0)
# pred_data = np.mean(pred_data, axis=0)
# print(true_data.shape)  # 应该输出 (10, 64, 64)
# print(pred_data.shape)  # 应该输出 (10, 64, 64)

# 计算64*64网格中每个点r2和rmse
# r2_matrix = np.zeros((T, H, W))
# rmse_matrix = np.zeros((T, H, W))
#
# for i in range(T):
#     for j in range(H):
#         for k in range(W):
#             # 提取当前点的真值和预测值的时间序列
#             true_vals = true_data[:, j, k]
#             pred_vals = pred_data[:, j, k]
#
#             # 计算当前点的r2
#             # r2 = r2_score(true_vals, pred_vals)
#
#             # 计算当前点的rmse
#             rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
#
#             # r2_matrix[i, j, k] = r2
#             rmse_matrix[i, j, k] = rmse
#


print(r2_matrix.shape)  # 应该输出 (10, 64, 64)
print(rmse_matrix.shape)  # 应该输出 (10, 64, 64)
print("r2_matrix:", r2_matrix[0])
print("rmse_matrix:", rmse_matrix[0])

results = np.zeros((H, W))
# 统计每个点累计10天中满足r2>0.97且rmse<0.1的天数
for i in range(H):
    for j in range(W):
        count = 0
        for k in range(T):
            if r2_matrix[k, i, j] >= 0.88 and rmse_matrix[k, i, j] <= 0.7:
                count += 1
        results[i, j] = count

print(results)
print(results.shape)
# 保存结果
np.save("npys/new_R2_0.88&RMSE_0.7.matrix.npy", results)
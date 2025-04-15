import numpy as np
import os
model_name = 'SSTPredictor'
data_path_true = f'D:/Datasets/result/20250414/reverse_true_sst_{model_name}_SCS.npy'
data_path_pred = f'D:/Datasets/result/20250414/reverse_pred_sst_{model_name}_SCS.npy'
mask_path = 'D:/GraduationThesis/codes/preData/maskLand=0.npy'
def read_npy(file_path):
    return np.load(file_path)

# 计算 MAE
def mean_absolute_error(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    return np.mean(np.abs(masked_y_true - masked_y_pred))

# 计算 RMSE
def root_mean_squared_error(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    return np.sqrt(np.mean((masked_y_true - masked_y_pred) ** 2))

# 计算 R²
def r2_score(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    ss_res = np.sum((masked_y_true - masked_y_pred) ** 2)
    ss_tot = np.sum((masked_y_true - np.mean(masked_y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan  # 避免除零错误

# 主函数
def calculate(true_file_path, pred_file_path, mask_file_path, output_file_path):
    # 读取数据
    try:
        y_true = read_npy(true_file_path)  # 形状 (batchsize, 10, 1, 64, 64)
        y_pred = read_npy(pred_file_path)  # 形状 (batchsize, 10, 1, 64, 64)
        mask = read_npy(mask_file_path)    # 形状 (64, 64)，值为 0 或 1
    except FileNotFoundError:
        print("文件路径错误，请检查路径!")
        return
    except ValueError as e:
        print(f"数据读取错误: {e}")
        return

    # 检查 mask 形状是否匹配
    if mask.shape != (64, 64):
        print("错误: mask 形状不匹配 (64,64)，请检查文件!")
        return
    # 将 mask 扩展为 (2040, 64, 64) 的形状
    mask = np.expand_dims(mask, axis=0)  # 形状变为 (1, 64, 64)
    mask = np.repeat(mask, y_true.shape[0], axis=0)  # 广播为和y_true_day的维度一致 (batchsize, 64, 64)
    print(mask.shape)
    # 初始化存储每天平均指标的列表
    mae_day_avg = []
    rmse_day_avg = []
    r2_day_avg = []

    # 计算每一天的平均指标
    days = list(range(y_true.shape[1]))  # 翻转天数顺序
    for day in days:
        # 提取当前天的数据
        y_true_day = y_true[:, day, :, :]  # (batchsize, 64, 64)
        y_pred_day = y_pred[:, day, :, :]  # (batchsize, 64, 64)
        # print(y_true_day.shape)
        # 计算所有 batch 的均值，应用 mask
        mae = mean_absolute_error(y_true_day, y_pred_day, mask)
        rmse = root_mean_squared_error(y_true_day, y_pred_day, mask)
        r2 = r2_score(y_true_day, y_pred_day, mask)

        # 存储每天的平均指标
        mae_day_avg.append(mae)
        rmse_day_avg.append(rmse)
        r2_day_avg.append(r2)

    # 将结果保存到文本文件并打印
    with open(output_file_path, "w", encoding="utf-8") as f:  # 指定 UTF-8 编码
        for i in range(len(mae_day_avg)):
            day_num = i + 1
            result_line = f"Day {day_num}: MAE: {mae_day_avg[i]:.4f}, RMSE: {rmse_day_avg[i]:.4f}, R²: {r2_day_avg[i]:.4f}\n"
            f.write(result_line)  # 写入文件
            print(result_line.strip())  # 同时打印到控制台
# 提供三个文件：
# true和pred文件：（batchsize，10,1,64,64）；
# mask文件：（64,64）；
# output_file_path：结果输出打印的文件.txt格式
calculate(data_path_true,
                   data_path_pred, mask_path,
                  'results.txt')
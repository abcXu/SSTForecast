import numpy as np
import numpy.ma as ma

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/reverse/reverse_pred_MoDeRNN_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/reverse/reverse_true_MoDeRNN_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/normalize/normalize_pred_MoDeRNN_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/normalize/normalize_true_MoDeRNN_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/reverse/reverse_pred_ConvLSTM_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/reverse/reverse_true_ConvLSTM_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/normalize/normalize_pred_ConvLSTM_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/normalize/normalize_true_ConvLSTM_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/reverse/reverse_pred_PredRNN_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/reverse/reverse_true_PredRNN_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/normalize/normalize_pred_PredRNN_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/normalize/normalize_true_PredRNN_Bohai.npy'

# pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/reverse/reverse_pred_CBAM-ConvLSTM_Bohai.npy'
# true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/reverse/reverse_true_CBAM-ConvLSTM_Bohai.npy'

pred_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/normalize/normalize_pred_CBAM-ConvLSTM_Bohai.npy'
true_path = 'F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/normalize/normalize_true_CBAM-ConvLSTM_Bohai.npy'


pred = np.load(pred_path).reshape(-1, 10, 64, 64)
true = np.load(true_path).reshape(-1, 10, 64, 64)

# 创建掩码数组(渤海+黄海)
mask = np.load('F:/实验数据/mask_array/mask.npy')

# 使用广播机制将掩码数组扩展到与真实值和预测值相同的维度
mask_broadcasted = np.broadcast_to(mask, (true.shape[0], true.shape[1], true.shape[2], true.shape[3]))

# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
true_values_masked = ma.masked_array(true, mask_broadcasted)
pred_values_masked = ma.masked_array(pred, mask_broadcasted)

true_masked_data = ma.filled(true_values_masked, 0)  # 将掩码为True的数据置为0
pred_masked_data = ma.filled(pred_values_masked, 0)  # 将掩码为True的数据置为0

#
# true_masked_data = ma.filled(true_values_masked, -999.9999)  # 将掩码为True的数据置为-999.9999
# pred_masked_data = ma.filled(pred_values_masked, -999.9999)  # 将掩码为True的数据置为-999.9999


np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/reverse/masked_data/true_masked_data.npy', true_masked_data)
np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/reverse/masked_data/pred_masked_data.npy', pred_masked_data)

# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/normalize/masked_data/normalize_true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/MoDeRNN/normalize/masked_data/normalize_pred_masked_data.npy', pred_masked_data)

# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/reverse/masked_data/true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/reverse/masked_data/pred_masked_data.npy', pred_masked_data)

np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/normalize/masked_data/normalize_true_masked_data.npy', true_masked_data)
np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ConvLSTM/normalize/masked_data/normalize_pred_masked_data.npy', pred_masked_data)

# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/reverse/masked_data/true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/reverse/masked_data/pred_masked_data.npy', pred_masked_data)
# #
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/normalize/masked_data/normalize_true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/PredRNN/normalize/masked_data/normalize_pred_masked_data.npy', pred_masked_data)

# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/reverse/masked_data/true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/reverse/masked_data/pred_masked_data.npy', pred_masked_data)

# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/normalize/masked_data/normalize_true_masked_data.npy', true_masked_data)
# np.save('F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/normalize/masked_data/normalize_pred_masked_data.npy', pred_masked_data)



print(true_masked_data.shape, pred_masked_data.shape)

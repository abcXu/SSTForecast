import numpy as np
import numpy.ma as ma
import netCDF4 as nc


name = ['ConvLSTM', 'TCTN', 'SwinLSTM', 'DatLSTM']

# origin_data  = np.load("E:/渤海海冰海温预处理数据/BoHai_sst_data/每天样本均值填充数据_保留四位小数_未归一化/sst_tianchong_mean_round_4.npy")

origin_data  = np.load("G:/SwinLSTM-main/Mask/每天样本均值填充数据_保留四位小数_未归一化/sst_tianchong_mean_round_4.npy")

# true_masked_data = np.load('F:/BoHaiSea_sic/'+name+'/reverse/reverse_true_'+name+'_Bohai.npy')
# true_masked_data = np.array(nc.Dataset('task/ST-GCFN_2012.nc')['data'])
# true_masked_data = np.load('E:/大论文实验数据/'+name[4]+'/reverse/reverse_true_'+name[4]+'_Bohai_sst.npy').reshape(-1, 10, 64, 64)
# true_masked_data = np.load('F:/实验数据(考虑瑞年_何老师)/experiment_result/'+name[6]+'/reverse/reverse_true_'+name[6]+'_Bohai_sst.npy').reshape(-1, 10, 64, 64)
true_masked_data = np.load('F:/实验数据(考虑瑞年_何老师)/experiment_result/'+name[6]+'/reverse/reverse_true_'+name[6]+'_Bohai_sst.npy').reshape(-1, 10, 64, 64)

# path + name +'/20240425/reverse_pred_'+name+'_BoHai_sst.npy'
mask = np.load('F:/会议_实验数据/mask_array/mask.npy')

# 使用广播机制将掩码数组扩展到与真实值和预测值相同的维度
origin_mask_broadcasted = np.broadcast_to(mask, (origin_data.shape[0], origin_data.shape[1], origin_data.shape[2]))
# true_mask_broadcasted = np.broadcast_to(mask, (true_masked_data.shape[0], true_masked_data.shape[1], true_masked_data.shape[2]))
true_mask_broadcasted = np.broadcast_to(mask, (true_masked_data.shape[0], true_masked_data.shape[1], true_masked_data.shape[2], true_masked_data.shape[3]))

# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
origin_values_masked = ma.masked_array(origin_data, origin_mask_broadcasted)
true_values_masked = ma.masked_array(true_masked_data, true_mask_broadcasted)

origin_masked_data = ma.filled(origin_values_masked, 0)  # 将掩码为True的数据置为0
true_data = ma.filled(true_values_masked, 0)  # 将掩码为True的数据置为0

for i in range(origin_data.shape[0]):
    true = np.round(true_data[0, 0, ...], 4)  #如果元素比较不相同，则说明该数据为下一个年份
    origin = np.round(origin_masked_data[i, ...], 4)
    # 比较有效元素
    if np.array_equal(origin , true):
        print(i)




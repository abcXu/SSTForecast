import numpy as np
import matplotlib.pyplot as plt

model_name = 'SSTPredictor'
# 定义文件路径
data_path_true = f'D:/Datasets/result/20250414/reverse_true_sst_{model_name}_SCS.npy'
data_path_pred = f'D:/Datasets/result/20250414/reverse_pred_sst_{model_name}_SCS.npy'
mask_path = 'D:/GraduationThesis/codes/preData/maskLand=0.npy'
def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def R_2(pred, true):
    u = np.mean(pred)
    v = np.mean(true)
    ssr = np.sum((pred - u) * (true - v))
    sst = np.sum((true - v) * (true - v))
    return ssr / sst

def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def getMeanMSE(trues, preds):
    mse_list = []
    for day in range(trues.shape[1]):
        mse = MSE(preds[:, day, ...], trues[:, day, ...])
        mse_list.append(mse)
    return mse_list
def getMeanR_2(trues, preds):
    r_2_list = []
    for day in range(trues.shape[1]):
        r_2 = R_2(preds[:, day, ...], trues[:, day, ...])
        r_2_list.append(r_2)
    return r_2_list
def getMeanMAE(trues, preds):
    '''

    :param trues: [N,T,H,W]
    :param preds: [N,T,H,W]
    :return: list [T]
    '''
    mae_list = []
    for day in range(trues.shape[1]):
        mae = MAE(preds[:, day, ...], trues[:, day, ...])
        mae_list.append(mae)
    return mae_list

###############################################################
def MAE_masked(pred, true, mask):
    '''

    :param pred: [N,H,W]
    :param true: [N,H,W]
    :param mask: [H,w]
    :return: MAE
    '''
    total = mask.sum()*pred.shape[0]
    masked_diff = np.abs(pred - true) * mask

    mae_masked = masked_diff.sum() / total
    # 返回保留三位小数的结果
    return round(mae_masked, 3)



def MSE_masked(pred, true, mask):
    total = mask.sum() * pred.shape[0]
    masked_diff = ((pred - true) ** 2) * mask
    mae_masked = masked_diff.sum() / total
    return round(mae_masked, 3)

def R_2_masked(pred, true, mask):
    total = mask.sum() * pred.shape[0]
    masked_pred = pred * mask
    masked_true = true * mask
    u = masked_pred.sum() / total
    v = masked_true.sum() / total
    ssr = ((masked_pred - u) * (masked_true - v)).sum()
    sst = ((masked_true - v) * (masked_true - v)).sum()
    r2_masked = ssr / sst
    return round(r2_masked, 3)

def getMeanMaskedMAE(trues, preds, mask):
    mae_list = []
    for day in range(trues.shape[1]):
        mae = MAE_masked(preds[:, day], trues[:, day], mask)
        mae_list.append(mae)
    return mae_list

def getMeanMaskedMSE(trues, preds, mask):
    mse_list = []
    for day in range(trues.shape[1]):
        mse = MSE_masked(preds[:, day], trues[:, day], mask)
        mse_list.append(mse)
    return mse_list

def getMeanMaskedR2(trues, preds, mask):
    r2_list = []
    for day in range(trues.shape[1]):
        r2 = R_2_masked(preds[:, day], trues[:, day], mask)
        r2_list.append(r2)
    return r2_list
#####################################################
if __name__ == '__main__':
    # preds = np.load(data_path_pred)
    # trues = np.load(data_path_true)
    # mae_list = getMeanMAE(trues, preds)
    # print("mae:",mae_list)
    # r2_list = getMeanR_2(trues, preds)
    # print("r2:",r2_list)
    # mse_list = getMeanMSE(trues, preds)
    # print("mse:",mse_list)
    # mask = np.load(mask_path)
    preds = np.load(data_path_pred)  # shape: [N, T, H, W]
    trues = np.load(data_path_true)
    mask = np.load(mask_path)  # shape: [H, W]

    mae_list = getMeanMaskedMAE(trues, preds, mask)
    print("Masked MAE:", mae_list)

    mse_list = getMeanMaskedMSE(trues, preds, mask)
    print("Masked MSE:", mse_list)

    r2_list = getMeanMaskedR2(trues, preds, mask)
    print("Masked R²:", r2_list)



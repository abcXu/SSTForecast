import os
from datetime import datetime

import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils.utils import compute_metrics, visualize

scaler = amp.GradScaler()

def train(args, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()

    num_batches = len(train_loader)
    losses = []

    for batch_idx, (inputs_sst,targets_sst,inputs_ssh,targets_ssh) in enumerate(train_loader):
        optimizer.zero_grad()
        # inputs / targets :[b, T, C, H, W]
        inputs_sst,targets_sst,inputs_ssh,targets_ssh = map(lambda x: x.float().to(args.device), [inputs_sst,targets_sst,inputs_ssh,targets_ssh])

        assert targets_sst.shape[1] == targets_ssh.shape[1],\
            "targets_sst and targets_ssh should have the same length"

        with autocast():
            if args.model == 'SSTPredictor':
                output_sst = model(inputs_sst,inputs_ssh)
            loss = criterion(output_sst, targets_sst)

        # 将损失进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        # 每log_train打印一次损失
        if batch_idx and batch_idx % args.log_train == 0:
            logger.info(f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f}')

    return np.mean(losses)

def valid(args, logger, epoch, model, valid_loader, criterion,cache_dir):
    model.eval()
    num_batches = len(valid_loader)
    losses_sst, mses_sst, ssims_sst = [], [], []

    for batch_idx, (inputs_sst, targets_sst, inputs_ssh, targets_ssh) in enumerate(valid_loader):

        with torch.no_grad():
            inputs_sst, targets_sst, inputs_ssh, targets_ssh = map(lambda x: x.float().to(args.device), [inputs_sst, targets_sst, inputs_ssh, targets_ssh])


            if args.model == 'SSTPredictor':
                outputs_sst = model(inputs_sst,inputs_ssh)

            losses_sst.append(criterion(outputs_sst, targets_sst).item())

            assert inputs_sst.shape[1]== inputs_ssh.shape[1],\
                "inputs_sst and inputs_ssh should have the same length"


            mse_sst, ssim_sst = compute_metrics(outputs_sst, targets_sst)

            mses_sst.append(mse_sst)
            ssims_sst.append(ssim_sst)

            if batch_idx and batch_idx % args.log_valid == 0:
                logger.info(
                    f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses_sst):.6f} MSE:{mse_sst:.4f} SSIM:{ssim_sst:.4f}')

                visualize(inputs_sst, targets_sst, outputs_sst, epoch, batch_idx, cache_dir, f'SST epoch:{epoch} batch:{batch_idx}')

    return np.mean(losses_sst), np.mean(mses_sst), np.mean(ssims_sst)


def test(args, logger, epoch, model, test_loader, criterion, cache_dir):
    model.eval()
    num_batches = len(test_loader)
    # 评价指标
    losses_sst, mses_sst, ssims_sst = [], [], []

    #  用于记录预测值以及真实值
    total_pred_sst = []
    total_true_sst = []

    for batch_idx, (inputs_sst, targets_sst, inputs_ssh, targets_ssh) in enumerate(test_loader):

        with torch.no_grad():
            inputs_sst, targets_sst, inputs_ssh, targets_ssh = map(lambda x: x.float().to(args.device), [inputs_sst, targets_sst, inputs_ssh, targets_ssh])

            if args.model == 'SSTPredictor':
                outputs_sst = model(inputs_sst,inputs_ssh)
                # outputs_sst = model(inputs_sst)
            # 记录sst的loss
            losses_sst.append(criterion(outputs_sst, targets_sst).item())
            assert inputs_sst.shape[1]== inputs_ssh.shape[1],\
                "inputs_sst and inputs_ssh should have the same length"

            mse_sst, ssim_sst = compute_metrics(outputs_sst, targets_sst)

            mses_sst.append(mse_sst)
            ssims_sst.append(ssim_sst)

            # 将每个批次的预测值和真实值添加到总列表中
            # [B,T,1,H,W] --> [B,T,H,W]
            sample_true_sst = (targets_sst.cpu().numpy()).squeeze(axis=2)  # (8,10,64,64)
            sample_pred_sst = (outputs_sst.cpu().numpy()).squeeze(axis=2)  # (8,10,64,64)

            # 将预测值和对应的真实值保存
            for b in range(args.test_batch_size):
                total_true_sst.append(sample_true_sst[b])
                total_pred_sst.append(sample_pred_sst[b])

        if batch_idx and batch_idx % args.log_valid == 0:
            logger.info(
                f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses_sst):.6f} MSE:{mse_sst:.4f} SSIM:{ssim_sst:.4f}')

    # 真实数据
    true_array_sst = np.array(total_true_sst)
    # 预测数据
    pred_array_sst = np.array(total_pred_sst)
    print('true_array_sst shape:', true_array_sst.shape)
    print('pred_array_ssh shape:', pred_array_sst.shape)
    # 将数据进行保存
    test_data_save_dir = args.test_data_save_dir
    # 获取当前日期
    current_date = datetime.now().strftime("%Y%m%d")
    # 创建保存路径
    save_path = test_data_save_dir + '{}/'.format(current_date)
    # 确保路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存真实数据以及预测数据
    print("Saved data path: ",save_path + f"normalized_true_sst_{args.model}_SCS.npy")
    np.save(
        save_path + f"normalized_true_sst_{args.model}_SCS.npy",
        true_array_sst)
    np.save(
        save_path + f"normalized_pred_sst_{args.model}_SCS.npy",
        pred_array_sst)

    # 返回每一个指标的均值
    return np.mean(losses_sst), np.mean(mses_sst), np.mean(ssims_sst)


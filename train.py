import torch
import torch.nn as nn
from configs import get_args
from functions import train, valid
from torch.utils.data import DataLoader
from data_provider.data_loader import SCSDataset
from models.SSTPMnew import SSTPredictor
from utils.utils import *

def setup(args):

    if args.model == 'SSTPredictor':
        model = SSTPredictor(input_dim=args.input_channels,hidden_dim=args.hidden_dim, d_model=args.d_model,\
                             nhead=args.nhead, in_len=args.num_frames_input,pred_len=args.num_frames_output, fusion_mode=args.fusion_mode).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    train_loader = DataLoader(
        SCSDataset(args, 'train'),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    valid_loader = DataLoader(
        SCSDataset(args, 'valid'),
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    return model, criterion, optimizer, train_loader, valid_loader


def main():
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, train_loader, valid_loader = setup(args)

    train_losses, valid_losses = [], []

    best_metric = (0, float('inf'), float('inf'))
    print("_____________________training________________________")
    for epoch in range(args.epochs):
        # 记录开始时间
        start_time = time.time()
        train_loss = train(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)

        if (epoch + 1) % args.epoch_valid == 0:

            valid_loss, mse, ssim = valid(args, logger, epoch, model, valid_loader, criterion, cache_dir)

            valid_losses.append(valid_loss)

            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)

            # 判断是否出现了最优的mse，是则将当前epoch的模型进行保存
            if mse < best_metric[1]:
                # 保存模型
                torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict_{args.model}')
                # 以元组的形式记录最佳指标
                best_metric = (epoch, mse, ssim)

            # 添加日志
            logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')
        # 打印一轮训练用掉的时间
        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    main()

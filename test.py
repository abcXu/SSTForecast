import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from configs import get_args
from data_provider.data_loader import SCSDataset
from models.SSTPM import SSTPredictor
from functions import test
from utils.utils import set_seed, make_dir, init_logger

if __name__ == '__main__':
    # 获取参数配置对象
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    # 初始化模型
    # model = SSTPredictor(input_dim=args.input_channels, hidden_dim=args.hidden_dim, d_model=args.d_model,\
    #                      nhead=args.nhead,in_len=args.num_frames_input,pred_len=args.num_frames_output, fusion_mode=args.fusion_mode).to(args.device)
    # model = SSTPredictorNoCA(input_dim=args.input_channels, hidden_dim=args.hidden_dim, d_model=args.d_model,\
    #                  in_len=args.num_frames_input,pred_len=args.num_frames_output).to(args.device)
    model = SSTPredictor(input_dim=args.input_channels, hidden_dim=args.hidden_dim, d_model=args.d_model, nhead=args.nhead, pred_len=args.num_frames_output, fusion_mode="mlp").to(args.device)
    criterion = nn.MSELoss()

    # 设置test_loader
    test_loader = DataLoader(
        SCSDataset(args,'test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    # 加载模型
    model.load_state_dict(torch.load(f'{model_dir}/trained_model_state_dict_{args.model}_mlp'))
    print("model_path: ", f'{model_dir}/trained_model_state_dict_{args.model}_mlp')
    # 记录开始时间
    start_time = time.time()

    _, mse, ssim = test(args, logger, 0, model, test_loader, criterion, cache_dir)
    # 打印结果
    print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
    # 打印训练耗时
    print(f'Time usage per epoch: {time.time() - start_time:.0f}s')


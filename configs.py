# parameters setting of the project
import argparse
# 数据路径
data_path = r'./data/all_data'
def get_args():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)

    # Setup parameters
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    # dataset parameters
    parser.add_argument('--data_path',default=data_path,help='path of dataset')
    parser.add_argument('--num_frames_input', default=10, type=int, help='Input sequence length')
    parser.add_argument('--num_frames_output', default=10, type=int, help='Output sequence length')
    parser.add_argument('--input_size', default=(64, 64), help='Input resolution')

    # model parameters
    parser.add_argument('--model', default='SSTPredictor', type=str, choices=['SSTPredictor','SSTPredictorWithoutSSH','SSTPredictorNoCA'],
                        help='Model type')
    parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--hidden_dim', default=64, type=int, help='GRU2D hid dimension')
    parser.add_argument('--d_model', default=256, type=int, help='CrossAttention Layer embedding dimension')
    parser.add_argument('--nhead', default=8, type=int, help='Number of heads in attention')
    parser.add_argument('--fusion_mode',default="mlp",type=str,help='fusion mode in CrossAttention Layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')

    # Training parameters
    parser.add_argument('--train_batch_size', default=8, type=int, help='Batch size for training')
    parser.add_argument('--valid_batch_size', default=8, type=int, help='Batch size for validation')
    parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--epoch_valid', default=1, type=int)
    parser.add_argument('--log_train', default=2, type=int)
    parser.add_argument('--log_valid', default=2, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--data_split_ratio',default=[0.7,0.1,0.2],help='the ratio of train valid test')
    # predict
    parser.add_argument('--test_data_save_dir',default='./results/predict/')
    parser.add_argument('--data_name', type=str, default='scs',choices=['Bohai_sic,scs'])
    args = parser.parse_args()

    return args

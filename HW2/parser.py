from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV HW2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='Hw2_data/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default= 0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=70, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=16, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=5e-4, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='model_best.pth.tar', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=9)
    parser.add_argument('--model', default='Net', type=str)

    args = parser.parse_args()

    return args

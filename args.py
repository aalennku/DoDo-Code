import argparse
import torch
import os
import shutil

def get_args(task=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=8, help='length of codeword')
    parser.add_argument('--seed', type=int, default=7777777, help='random seed (not guarentee)')
    parser.add_argument('--dimension', type=int, default=64, help='dimension of embedding vector')
    parser.add_argument('--epochs', type=int, default=4, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='label of gpu')
    parser.add_argument('--save-path', type=str, default='./',help='path to save the embed model or sth')
    parser.add_argument('--constraint',type=bool,default=False,help='flag to apply constraint or not')
    parser.add_argument('--eccsym',type=float,default=0.01,help='num of ecc symbols if use rs code')

    if task == 'test_codeword':
        parser.add_argument('--number', type=int, default=100000,
                        help='number of codewords with ids in test_codeword.py')
        parser.add_argument('--num-neighbors', type=int, default=5,
                            help='number of neighbors to search')
        
    if task == 'test_sequence':
        parser.add_argument('--number', type=int, default=1000,
                        help='number of sequence with ids in test_sequence.py')
        parser.add_argument('--num-neighbors', type=int, default=5,
                            help='number of neighbors to search')
        parser.add_argument('--max-err', type=int, default=7,
                            help='max accepted error in the DFS')
        parser.add_argument('--err-rate', type=float, default=0.01,
                            help='error rate in test_sequence.py')

    args = parser.parse_args()
    args.padded_length = args.length + 4
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.num_pieces = int(150/args.length)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    return args
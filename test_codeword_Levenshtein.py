from args import get_args
import torch
from model import load_embed_model
import os
from utils import ids_channel
import numpy as np
from torch.nn.functional import one_hot
import Levenshtein as L
from scipy.spatial import KDTree
from tqdm import tqdm
import time


def main(args):
    codeword_save_path = os.path.join(args.save_path,'codeword.npz')

    codeword = np.load(codeword_save_path)['codeword']

    retrieved_seq_list = []
    groundtruth_list = []
    for _ in tqdm(range(args.number)):
        while True:
            profile = np.zeros(args.padded_length).astype(int).tolist()
            random_number = np.random.rand()
            if random_number < 1/3:
                profile[np.random.randint(args.padded_length)] = np.random.randint(1,4)
            elif random_number < 2/3:
                profile[np.random.randint(args.padded_length)] = np.random.randint(4,8)
            else:
                profile[np.random.randint(args.padded_length)] = 8
            
            rand_idx = np.random.randint(len(codeword))
            seq = codeword[rand_idx]
            seq_retrieved = np.array(ids_channel(seq, profile))
            if L.distance(seq,seq_retrieved) == 1:
                break
            
        seq_retrieved = seq_retrieved[:args.padded_length]
        seq_retrieved = np.concatenate([seq_retrieved,
                                       [4]*(args.padded_length-len(seq_retrieved))]).astype(np.int64)
        retrieved_seq_list.append(seq_retrieved)
        groundtruth_list.append(seq)


    start_time = time.time()
    corrected_seq = []
    retrieved_seq_list = np.array(retrieved_seq_list)
    for idx_seq,item in enumerate(tqdm(retrieved_seq_list)):
        for codeword_idx, seq_codeword in enumerate(codeword):
            d = L.distance(item,seq_codeword)
            item = item[item!=4]
            if d<= 1:
                corrected_seq.append(seq_codeword)
                break
        else:
            corrected_seq.append(codeword[0])
    time_used = time.time()-start_time
    print('time used: {}'.format(time_used))

    cnt = 0
    err_cnt = 0
    for seq_a, seq_b in zip(groundtruth_list,corrected_seq):
        cnt += 1
        if not np.array_equal(seq_a,seq_b):
            err_cnt += 1
    print('Brute-force segment correcting: , err: {}, tot: {}'.format(err_cnt, cnt))

    with open(os.path.join(args.save_path,'result_segment_correcting_brute_froce.txt'),'a') as f:
        f.write('Brute-force segment correcting: , err: {}, tot: {}\n'.format(err_cnt, cnt))
        f.write('time used: {}\n=============\n'.format(time_used))


if __name__ == '__main__':
    args = get_args('test_codeword')
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    main(args)
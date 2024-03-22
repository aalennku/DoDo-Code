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
    embed_model = load_embed_model(args)
    codeword_save_path = os.path.join(args.save_path,'codeword.npz')

    codeword = np.load(codeword_save_path)['codeword']
    codeword_emb = np.load(codeword_save_path)['emb']

    kd_codeword = KDTree(codeword_emb)

    retrieved_seq_list = []
    retrieved_seq_list_nopad = []
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
            
            seq = codeword[_%len(codeword)]
            seq_retrieved = np.array(ids_channel(seq, profile))
            if L.distance(seq,seq_retrieved) == 1:
                break
            
        seq_retrieved = seq_retrieved[:args.padded_length]
        retrieved_seq_list_nopad.append(seq_retrieved[:])
        seq_retrieved = np.concatenate([seq_retrieved,
                                       [4]*(args.padded_length-len(seq_retrieved))]).astype(np.int64)
        retrieved_seq_list.append(seq_retrieved)
        groundtruth_list.append(seq)
    
    corrected_seq = []
    num_neighbors = args.num_neighbors
    
    retrieved_emb_list = []
    retrieved_seq_list = np.array(retrieved_seq_list)
    start_time = time.time()
    for _ in range(len(retrieved_seq_list)//1024+int(len(retrieved_seq_list)%1024!=0)):
        emb = embed_model(one_hot(torch.tensor(retrieved_seq_list[_*1024:(_+1)*1024]), num_classes=5).transpose(-1,-2).to(torch.float).to(device))
        retrieved_emb_list.append(emb.detach().cpu().numpy())
    retrieved_emb_list = np.concatenate(retrieved_emb_list,axis=0)
    print('emb time: {}\n'.format(time.time()-start_time))

    for idx_seq,item in enumerate(tqdm(retrieved_emb_list)):
        d,idx = kd_codeword.query(item,num_neighbors)
        min_d = 99999999
        shot_idx = 99999999
        if num_neighbors != 1:
            retrieved_seq_nopad = retrieved_seq_list_nopad[idx_seq]
            for seq_idx in idx:
                d = L.distance(codeword[seq_idx],retrieved_seq_nopad)
                if d < min_d:
                    min_d = d
                    shot_idx = seq_idx
        else:
            shot_idx = idx
        corrected_seq.append(codeword[shot_idx])
    time_used = time.time()-start_time
    print('time used: {}'.format(time_used))


    cnt = 0
    err_cnt = 0
    for seq_a, seq_b in zip(groundtruth_list,corrected_seq):
        cnt += 1
        if not np.array_equal(seq_a,seq_b):
            err_cnt += 1
    print('Segment correcting, num neighbors: {}, err: {}, tot: {}'.format(args.num_neighbors, err_cnt, cnt))

    with open(os.path.join(args.save_path,'result_segment_correcting.txt'),'a') as f:
        f.write('Segment correcting, num neighbors: {}, err: {}, tot: {}\n'.format(args.num_neighbors, err_cnt, cnt))
        f.write('time used: {}\n=============\n'.format(time_used))


if __name__ == '__main__':
    args = get_args('test_codeword')
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    main(args)
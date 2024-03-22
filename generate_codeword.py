from args import get_args
import torch
from model import load_embed_model
import os
from utils import brute_full_word_generator
import numpy as np
from torch.nn.functional import one_hot
from sklearn.decomposition import PCA
import Levenshtein as L

def get_criterion(total_emb):
    pca = PCA()
    pca.fit(total_emb)
    criterion = []
    for item in total_emb:
        weight = item.reshape(1,-1)@pca.components_.T@np.diag(pca.explained_variance_**(-1))@pca.components_@item.reshape(1,-1).T
        criterion.append(weight[0,0])
    criterion = np.array(criterion)
    return criterion

def pass_constraint(seq):
    flag = True
    K = 3
    seq = np.array(seq)
    if np.sum(seq>1)/len(seq) > 0.6:
        flag = False
    elif np.sum(seq>1)/len(seq) < 0.4:
        flag = False
    for _ in range(len(seq)-K):
        if len(set(seq[_:_+K])) == 1:
            flag = False
    return flag

def main(args):
    embed_model = load_embed_model(args)
    full_seq_list = brute_full_word_generator(args.length)
    full_seq_pad_list = np.concatenate([full_seq_list,
                                         np.array([[4]*(args.padded_length-args.length)]*full_seq_list.shape[0])],axis=1).astype(np.int64)

    full_seq_emb_list = []
    for _ in range(len(full_seq_pad_list)//1024): # full_seq_pad_list is int times of 1024
        emb = embed_model(one_hot(torch.tensor(full_seq_pad_list[_*1024:(_+1)*1024]), num_classes=5).transpose(-1,-2).to(torch.float).to(device))
        full_seq_emb_list.append(emb.detach().cpu().numpy())
    full_seq_emb = np.concatenate(full_seq_emb_list,axis=0)

    greedy_chozen_codeword = []
    greedy_chozen_idx = []
    available = np.array(list(range(4**args.length)))
    criterion = get_criterion(full_seq_emb)

    while np.sum(available!=-1) >= 1:
        print('Chozen: {}\t, Remain: {}.'.format(len(greedy_chozen_idx), np.sum(np.array(available)!=-1)))
        max_criterion = -1
        idx = None
        for _ in available:
            if _ == -1:
                continue
            c = criterion[_]
            if c > max_criterion:
                if not args.constraint:
                    max_criterion = c
                    idx = _
                elif pass_constraint(full_seq_list[_]):
                    max_criterion = c
                    idx = _
        if max_criterion == -1:
            break
        chozen = full_seq_list[idx]
        greedy_chozen_codeword.append(chozen[:])
        greedy_chozen_idx.append(idx)
        available[idx] = -1
        
        for _ in available:
            if _ == -1:
                continue
            distance_ = L.distance(chozen, full_seq_list[_])
            if distance_ <= 2:
                available[_] = -1

    greedy_chozen_emb = full_seq_emb[greedy_chozen_idx]
    print('We have generated {} codewords with the embed model.'.format(len(greedy_chozen_idx)))
    codeword_save_path = os.path.join(args.save_path,'codeword.npz')
    np.savez(codeword_save_path,codeword=greedy_chozen_codeword,idx=greedy_chozen_idx,emb=greedy_chozen_emb)

if __name__ == '__main__':
    args = get_args()
    global device
    device = args.device
    main(args)
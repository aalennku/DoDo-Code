import numpy as np
import random
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
from model import load_embed_model
from torch.nn.functional import one_hot
import torch
import Levenshtein as L
from scipy.spatial import KDTree
import os


class DODO_CODE():
    def __init__(self, length, padded_length, num_pieces, args):
        self.length = length
        self.padded_length = padded_length
        self.num_pieces = num_pieces
        self.embed_model = load_embed_model(args)
        self.args = args

        codeword_save_path = os.path.join(args.save_path,'codeword.npz')
        self.code = np.load(codeword_save_path)['codeword']
        self.codeword_emb = np.load(codeword_save_path)['emb']
        self.kd_codeword = KDTree(self.codeword_emb)
        print('CODE at rate {}/{}.'.format(np.log(len(self.code))/np.log(4),self.length))
    
    def random_info(self,seed=None):
        if not seed is None:
            random.seed(seed)
        return random.randint(0,len(self.code)**self.num_pieces-1)

    def encode(self,n):
        reference = []
        reference_idx = []
        if hasattr(n,'__len__'):
            reference = [self.code[idx] for idx in n]
            return np.concatenate(reference), n
        num_codewords = self.code.shape[0]
        for _ in range(self.num_pieces):
            codeword_idx = n % num_codewords
            n = n // num_codewords
            reference.append(self.code[codeword_idx])
            reference_idx.append(codeword_idx)
        return np.concatenate(reference), reference_idx
    
    def DFS_decode(self, read, root, leaf_node_list, level=0, num_neighbors=None, max_err=None):
        level += 1
        piece_list = []
        piece_list_nopad = []
        code_length = self.length
        if num_neighbors is None:
            num_neighbors = self.args.num_neighbors
        if max_err is None:
            max_err = self.args.max_err
        for _ in range(-1,2):
            piece = read[:code_length+_]
            piece_list_nopad.append(piece[:])
            piece = np.concatenate([piece,[4]*(self.padded_length-len(piece))]).astype(np.int64)
            piece_list.append(piece)
        piece_list = np.array(piece_list)
        piece_emb_list = self.embed_model(
                            one_hot(torch.tensor(piece_list)).transpose(-1,-2).to(torch.float).to(self.args.device))
        piece_emb_list = piece_emb_list.detach().cpu().numpy()

        corrected_piece_list = []
        corrected_distance = []
        corrected_idx = []
        for piece_idx, piece_emb in enumerate(piece_emb_list):
            d, neighbor_idx = self.kd_codeword.query(piece_emb, num_neighbors)
            min_d = 999999999
            shot_idx = 999999999
            if num_neighbors != 1:
                piece_nopad = piece_list_nopad[piece_idx]
                for seq_idx in neighbor_idx:
                    d = L.distance(self.code[seq_idx],piece_nopad)
                    if d < min_d:
                        min_d = d
                        shot_idx = seq_idx
            else:
                shot_idx = neighbor_idx
            corrected_piece_list.append(self.code[shot_idx])
            corrected_distance.append(min_d)
            corrected_idx.append(shot_idx)

        redo_flag = False
        if sorted(corrected_distance)[0] == 0:
            child = Node(level,parent=root)
            child.err = root.err
            child.err_trace = root.err_trace + [0]
            child.trace = root.trace+[corrected_piece_list[1]]
            child.trace_idx = root.trace_idx + [corrected_idx[1]]
            child.read_trace = root.read_trace+[piece_list_nopad[1]]
            if len(read)>self.length:
                self.DFS_decode(read[self.length:], child, leaf_node_list, level, num_neighbors, max_err)
            else:
                leaf_node_list.append(child)
                return
            redo_flag = True
            if len(child.children) != 0:
                for grandchild in child.children:
                    if grandchild.err_trace[-1] == 0:
                        redo_flag = False
        if sorted(corrected_distance)[0] != 0 or redo_flag:
            for idx, (piece, item, d, cidx) in enumerate(zip(piece_list_nopad,corrected_piece_list,corrected_distance,corrected_idx)):
                if redo_flag and d == 0:
                    continue
                child = Node(level, parent=root)
                child.err = root.err + d
                child.err_trace = root.err_trace +[d]
                child.trace = root.trace+[item]
                child.trace_idx = root.trace_idx + [cidx]
                child.read_trace = root.read_trace+[piece]
                if child.err > max_err:
                    leaf_node_list.append(child)
                    return
                else:
                    if len(read)>self.length-1+idx:
                        self.DFS_decode(read[self.length-1+idx:], child, leaf_node_list, level, num_neighbors, max_err)
                    else:
                        leaf_node_list.append(child)
                        return
                    
    def decode(self, read):
        root = Node('root')
        root.err = 0
        root.err_trace = []
        root.trace = []
        root.trace_idx = []
        root.read_trace = []
        leaf_node_list = []

        self.DFS_decode(read,root,leaf_node_list,self.args.num_neighbors,self.args.max_err)
        min_err = 7777777
        decoded_seq = None
        good_leaf = None
        for node in leaf_node_list:
            if np.sum(np.array(node.err_trace)>1)!=0:
                continue
            if node.err < min_err:
                min_err = node.err
                good_leaf = node
        if good_leaf is None:
            for node in leaf_node_list:
                if node.err < min_err:
                    min_err = node.err
                    good_leaf = node

        if good_leaf is None:
            decoded_seq = np.random.randint(4,size=self.length*self.num_pieces)
            decoded_idx_seq = np.random.randint(len(self.codeword), size=self.num_pieces)
        else:
            decoded_seq = np.concatenate(good_leaf.trace)
            decoded_idx_seq = good_leaf.trace_idx
        return decoded_seq, decoded_idx_seq, good_leaf




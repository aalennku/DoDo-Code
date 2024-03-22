import random
from tqdm import tqdm
from args import get_args
import torch
from DODO_CODE import DODO_CODE
import numpy as np
from utils import ids_channel
import os

def reference2read(reference):
    flag = True
    while flag:
        profile = np.zeros(len(reference)+args.length*5,dtype=np.int64)
        interval_start = 0
        interval_end = interval_start+args.length
        error_cnt = 0
        for idx in range(len(profile)):
            if np.random.rand() < args.err_rate:
                random_number = np.random.rand()
                if random_number < 1/3:
                    profile[idx] = np.random.randint(1,4)
                    error_cnt += 1
                elif random_number < 2/3:
                    profile[idx] = np.random.randint(4,8)
                    error_cnt += 1
                    interval_end += 1
                else:
                    profile[idx] = 8
                    error_cnt += 1
            if error_cnt > 1:
                break
            if idx == interval_end-1:
                interval_start = idx+1
                interval_end = interval_start+args.length
                error_cnt = 0
        else:
            flag = False

    read = np.array(ids_channel(reference,profile))
    return read, profile

def main(args):
    err_cnt = 0
    cnt = 0
    seg_err_cnt = 0
    seg_cnt = 0
    code = DODO_CODE(args.length, args.padded_length, args.num_pieces, args)
    for _ in tqdm(range(args.number)):
        random_array = code.random_info(seed=0)
        reference, reference_idx = code.encode(random_array)
        read, profile = reference2read(reference)
        decoded_sequence,decoded_idx, node = code.decode(read)

        decoded_sequence = decoded_sequence[:len(reference)]
        decoded_sequence = np.concatenate([decoded_sequence,np.random.randint(4,size=(len(reference)-len(decoded_sequence)))])
        
        cnt += 1
        seg_cnt += len(reference)/args.length
        if not np.array_equal(decoded_sequence, reference):
            err_cnt += 1

            reference_fold = reference.reshape(-1,args.length)
            decoded_sequence_fold = decoded_sequence.reshape(-1,args.length)
            for decode_segment, reference_segment in zip(decoded_sequence_fold, reference_fold):
                if not np.array_equal(decode_segment,reference_segment):
                    seg_err_cnt += 1
            print('err_cnt: {}, seg_err_cnt: {}'.format(err_cnt,seg_err_cnt))

    with open(os.path.join(args.save_path,'result_sequence_decoding_double_error_forbiden.txt'),'a') as f:
        f.write('Sequence err: {}, tot: {}, 1-acc: {}\n'.format(err_cnt, cnt, err_cnt/cnt*100))
        f.write('Segment err: {}, tot: {}, 1-acc: {}\n'.format(seg_err_cnt, seg_cnt, seg_err_cnt/seg_cnt*100))

    print('Sequence err: {}, tot: {}, 1-acc: {}\n'.format(err_cnt, cnt, err_cnt/cnt*100))
    print('Segment err: {}, tot: {}, 1-acc: {}\n'.format(seg_err_cnt, seg_cnt, seg_err_cnt/seg_cnt*100))


if __name__ == '__main__':
    args = get_args('test_sequence')
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    main(args)
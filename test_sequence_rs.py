import random
from tqdm import tqdm
from args import get_args
import torch
from DODO_CODE import DODO_CODE
import numpy as np
from utils import ids_channel
import os
import galois
from galois import ReedSolomon, Field

def reference2read(reference):
    profile = np.zeros(len(reference)+args.length*5,dtype=np.int64)
    for idx in range(len(profile)):
        if np.random.rand() < args.err_rate:
            random_number = np.random.rand()
            if random_number < 1/3:
                profile[idx] = np.random.randint(1,4)
            elif random_number < 2/3:
                profile[idx] = np.random.randint(4,8)
            else:
                profile[idx] = 8
    read = np.array(ids_channel(reference,profile))
    return read

def generate_random_with_RSECC(rsc, max_symbol, len_message):
    message = np.random.randint(0,max_symbol,size=len_message)
    message_ecc = np.array(rsc.encode(message))
    return message, message_ecc

def decode_RS(rsc, retrieved_message_ecc):
    message_recover = rsc.decode(retrieved_message_ecc)
    return np.array(message_recover)

def main(args):
    cnt = 0
    err_cnt = 0
    seg_err_cnt = 0
    seg_cnt = 0
    code = DODO_CODE(args.length, args.padded_length, args.num_pieces, args)
    max_symbol = len(code.code)
    max_symbol = galois.prev_prime(max_symbol)
    print('using {} of {} codewords'.format(max_symbol,len(code.code)))
    GF = Field(max_symbol)
    eccsym = int(np.round(args.eccsym * max_symbol))
    len_message = max_symbol-1-eccsym
    print('ECC ratio: {} {}'.format(eccsym/max_symbol, eccsym))
    rscodec = ReedSolomon(max_symbol-1,max_symbol-1-eccsym,field=GF)

    for _ in tqdm(range(100000//len_message)):
        message, message_ecc = generate_random_with_RSECC(rscodec,max_symbol=max_symbol,len_message=len_message)
        message = np.array(message)
        zeros = np.zeros(args.num_pieces - len(message_ecc) % args.num_pieces, dtype=np.int64)
        message_packet_pad = np.concatenate([message_ecc, zeros])
        message_packet_pad = message_packet_pad.reshape(-1,args.num_pieces)
        retrieved_packet_pad = []
        for message_one_sequence in tqdm(message_packet_pad):
            reference, reference_idx_seq = code.encode(message_one_sequence)
            read = reference2read(reference)
            decoded_sequence, decoded_idx_seq, node= code.decode(read)
            decoded_idx_seq = decoded_idx_seq[:len(reference_idx_seq)]
            decoded_idx_seq = np.concatenate([decoded_idx_seq,np.zeros(len(reference_idx_seq)-len(decoded_idx_seq),dtype=np.uint32)])
            retrieved_packet_pad.append(decoded_idx_seq)
        
        retrieved_message_ecc = np.array(retrieved_packet_pad).flatten()
        retrieved_message_ecc = retrieved_message_ecc[:len(message_ecc)]
        retrieved_message_ecc[retrieved_message_ecc>=max_symbol] = 7

        for a,b in zip(message_ecc[:len(message)], retrieved_message_ecc[:len(message)]):
            seg_cnt += 1
            if a!=b:
                seg_err_cnt += 1

        retrieved_message = decode_RS(rscodec,retrieved_message_ecc)

        for a,b in zip(message, retrieved_message):
            cnt += 1
            if a!=b:
                err_cnt += 1

        print(seg_cnt, seg_err_cnt)
        print(cnt, err_cnt)

    with open(os.path.join(args.save_path,'result_sequence_decoding_RS.txt'),'a') as f:
        f.write("eccsym: {}, max_sym:{}\n".format(eccsym, max_symbol))
        f.write('Segment err (ECC): {}, tot: {}, 1-acc: {}\n'.format(err_cnt, cnt, err_cnt/cnt*100))
        f.write('Segment err: err: {}, tot: {}, 1-acc: {}\n'.format(seg_err_cnt, seg_cnt, seg_err_cnt/seg_cnt*100))

    print('Segment err (ECC): {}, tot: {}, 1-acc: {}'.format(err_cnt, cnt, err_cnt/cnt*100))
    print('Segment err: err: {}, tot: {}, 1-acc: {}'.format(seg_err_cnt, seg_cnt, seg_err_cnt/seg_cnt*100))


if __name__ == '__main__':
    args = get_args('test_sequence')
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    main(args)
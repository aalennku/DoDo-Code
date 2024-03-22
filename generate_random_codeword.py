from args import get_args
from utils import brute_full_word_generator
import numpy as np
import Levenshtein as L


def main(args):
    full_seq_list = brute_full_word_generator(args.length)

    greedy_chozen_codeword = []
    greedy_chozen_idx = []
    available = np.array(list(range(4**args.length)))

    while np.sum(available!=-1) >= 1:
        print('Chozen: {}\t, Remain: {}.'.format(len(greedy_chozen_idx), np.sum(np.array(available)!=-1)))
        idx = np.random.choice(available[available!=-1])
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

    print('We have generated {} codewords randomly.'.format(len(greedy_chozen_idx)))
    with open('random/number_of_randomly_generated_codewords_{}.txt'.format(args.length),'a') as f:
        f.write('We have generated {} codewords randomly. seed: {}.\n'.format(len(greedy_chozen_idx),args.seed))

if __name__ == '__main__':
    args = get_args()
    global device
    device = args.device
    main(args)
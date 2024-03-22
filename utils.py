import numpy as np
import Levenshtein as L

# def ids_channel(s, profile):
#     ss = []
#     idx = 0
#     idx_profile = 0
#     while idx_profile < len(profile):
#         e = profile[idx_profile]
#         if idx >= len(s):
#             if e<=7 and e>3:
#                 ss.append(e%4)
#             else:
#                 break
#         else:
#             if e <= 3:
#                 aligo = np.array(s[idx])
#                 aligo = (aligo + e) % 4
#                 ss.append(aligo.tolist())
#             elif e<=7:
#                 ss.append(e%4)
#                 ss.append(s[idx])
#         idx += 1
#         idx_profile += 1
#     return ss

def ids_channel(s, profile):
    ss = []
    idx = 0
    profile = profile[:]
    idx_profile = 0
    while idx_profile < len(profile):
        e = profile[idx_profile]
        if idx >= len(s):
            if e<=7 and e>3:
                ss.append(e%4)
            else:
                break
        else:
            if e <= 3:
                aligo = s[idx]
                aligo = (aligo + e) % 4
                ss.append(aligo)
                idx += 1
            elif e<=7:
                ss.append(e%4)
            elif e == 8:
                idx += 1
        idx_profile += 1
    return ss

def generate_sample(length, padded_length, distance=1):
    ddd = -1
    while ddd != distance:
        seq_a = np.random.randint(0,4,size=length).astype(int)
        profile = np.zeros(padded_length).astype(int).tolist()
        # random_vec = np.random.rand(length).tolist()
        for _ in range(np.random.randint(1,5)):
            random_number = np.random.rand()
            if random_number < 1/3:
                profile[np.random.randint(padded_length)] = np.random.randint(1,4)
            elif random_number < 2/3:
                profile[np.random.randint(padded_length)] = np.random.randint(4,8)
            else:
                profile[np.random.randint(padded_length)] = 8
        seq_b = np.array(ids_channel(seq_a,profile))
        ddd = L.distance(seq_a,seq_b)
    return seq_a, seq_b

def brute_full_word_generator(length):
    if length == 1:
        return np.array([[0],[1],[2],[3]])
    short = brute_full_word_generator(length-1)
    codeword = []
    for _ in range(4):
        codeword.append(np.concatenate([np.array([[_]]*len(short)),short],axis=-1))
    return np.concatenate(codeword,axis=0)
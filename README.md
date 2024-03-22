### Train the embedding network for sequence length 8. 
    python3 train_embed_model.py --length 8 --seed 0 --gpu 0 --save-path='./8/0/'

### Generate the codewords of by the trained embedding network. 
    python3 generate_codeword.py --length 8 --seed 0 --gpu 0 --save-path='./8/0/'


### link the best run with most selected codewords to './8/best' by 'ln -s ./8/{SEED} ./8/best'
### Experiments on segment correcting
    python3 test_codeword.py --seed 0 --num-neighbors 5 --gpu 0 --length 8 --save-path="./8/best" 

### Experiments on sequence decoding
    python3 test_sequence.py --length 8 --seed 0 --gpu 0 --save-path='./8/best/' --number=10000

### Experiments on sequence decoding with an outer ECC of RS code. The ratio of ecc symbol is 2% in this case
    python3 test_sequence_rs.py --length 8 --seed 0 --gpu 0 --save-path='./8/best/' --eccsym=0.02


======================= Comparison \ Ablation study ===========================
### Generate codeword by random search. Results saved in './random'
    python3 generate_random_codeword.py --seed 0 --length 8

### Segment correcting by brute-force search. Mainly compares time complexity. 
    python3 test_codeword_Levenshtein.py --seed 0 --length=8 --save-path="./8/best"

### Sequence decoding which forbids more than one error in the same segment. 
    python3 test_sequence_double_error_forbidden.py --length 8 --seed 0 --gpu 0 --save-path='./8/best/' --number=10000

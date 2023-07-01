import numpy as np


# define input string
def Encode(data):
    # define universe of possible input values
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    # integer encode input data
    integer_encoded = []
    for char in data:
        if char in char_to_int:
            integer_encoded.append(char_to_int[char])
        else:
            integer_encoded.append(-1)
    # print(integer_encoded)

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        if value >= 0:
             letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


file_list = ['train_neg-original_data', 'train_pos-original_data']
for f_l in file_list:
    files = open(f_l+'.fasta').readlines()
    onehot_out = []
    for f in files:
        if not '>' in f:
            f = f.strip('\n')
            if len(f) > 1000:
                f = f[:1000]
            else:
                for i in range(len(f),1000):
                    f = f + 'X'
            seq = Encode(f)
            onehot_out.append(seq)
    np.save(f_l+'_onehot.npy', onehot_out)

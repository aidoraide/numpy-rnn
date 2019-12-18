import sys
import os
import pickle
import numpy as np
from train_rnn import token2printable, VOCAB, model_path, tokenize, Model

if not os.path.exists(model_path):
    exit(f'No model found in {model_path}. Run train_rnn.py to train a model.')

if len(sys.argv) == 1:
    exit('You must include the seed text to start text generation as a parameter')
seed = tokenize(sys.argv[1])

with open(model_path, 'rb') as f:
    rnn = pickle.load(f)

if __name__ == '__main__':
    yOut = rnn.forward(seed)
    scripture = [token2printable(t) for t in seed]
    while True:
        idx = np.random.choice(len(VOCAB), p=yOut[-1].ravel())
        c = VOCAB[idx]
        if c == '<VERSE_NUM>': break
        scripture.append(token2printable(c))
        yOut = rnn.forward([c])
    print()
    print((''.join(scripture)).strip())
    print()

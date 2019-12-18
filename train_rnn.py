import numpy as np
import math
import time
import re
import random
from collections import Counter

regex2punc = {
    r';': ' ; ',
    r':': ' : ',
    r',': ' , ',
    r'\.': ' . ',
    r'\?': ' ? ',
    r'!': ' ! ',
    r'\*': '',
}
data = open('bible.txt').read().lower()
data = data.split('End of the Project Gutenberg EBook of The King James Bible')[0] # cut out copyright
data = re.sub(r'[0-9]+:[0-9]+', ' <VERSE_NUM> ', data) # replace verse nums with tokens
# data = re.sub(r'\n\n', ' <NEW_LINE> ', data) # make double newlines a token
data = re.sub(r'\n', ' ', data) # remove single newlines
data = re.sub(r'\(.*\)', ' ', data) # remove brackets and anything inside
for regex, punc in regex2punc.items():
    data = re.sub(regex, punc, data)
data = [token for token in data.split(' ') if token != '']

# counter = Counter(data)
# data = [token if counter[token] > 1 else '<RARE>' for token in data]

# for i, c in enumerate(sorted(VOCAB, key=lambda c: counter[c])):
#     print(f"{c:15}: {counter[c]*100/len(data):5.2f}%")


VOCAB = sorted(list(set(data)))
c2i = {c:i for i, c in enumerate(VOCAB)}
i2c = VOCAB
def clean(c):
    return c

def onehot(c):
    idx = c2i[clean(c)]
    x = np.zeros((len(VOCAB), 1), dtype=np.int32)
    x[idx] = 1
    return x

def getPredictedChar(yOut):
    idx = np.argmax(yOut)
    return i2c[idx]


vnMajor = 1
vnMinor = 1
printableDict = {
    ';': ';',
    ':': ':',
    ',': ',',
    '.': '.',
    '?': '?',
    '!': '!',
    '*': '',
    '<NEW_LINE>': '\n\n',
}
def token2printable(token):
    if token == '<VERSE_NUM>':
        global vnMajor, vnMinor
        verseNum = f'\n\n{vnMajor}:{vnMinor}'
        if random.random() > 0.9:
            vnMajor, vnMinor = vnMajor + 1, 1
        else:
            vnMinor += 1
        return verseNum
    if token in printableDict:
        return printableDict[token]
    return ' ' + token

def softmax(x):
    shiftx = x - x.max()
    exps = np.exp(shiftx)
    return exps/exps.sum()

def dsoftmax(x):
    shiftx = x - x.max()
    exps = np.exp(shiftx)
    sumExps = exps.sum()
    return ((sumExps - exps) * exps) / (sumExps**2)

def L2(w):
    lamb = 1e-5
    return lamb * w**2

d = 40
v = len(VOCAB)
seq_len = 15
class Model:
    def __init__(self):
        # Each char has a d x 1 vector
        # Ht = tanh(Wh xi + Uh h(t-1) + bh)
        # yt = softmax(Wy h(t) + by)

        self.Wh = np.eye(d, v) + np.random.randn(d, v) / d 
        self.Uh = np.eye(d, d) + np.random.randn(d, d) / d
        self.Wy = np.eye(v, d) + np.random.randn(v, d) / d
        self.bh = np.zeros((d, 1))
        self.by = np.zeros((v, 1))
        self.mWh = np.zeros_like(self.Wh)
        self.mUh = np.zeros_like(self.Uh)
        self.mbh = np.zeros_like(self.bh)
        self.mWy = np.zeros_like(self.Wy)
        self.mby = np.zeros_like(self.by)
        self.h = [np.zeros((d, 1))]
        self.xs = []


    def resetInternal(self, full=False):
        self.h = [np.zeros((d, 1)) if full else self.h[-1]]
        self.xs = []


    def forward(self, batch):
        ys = [None] * len(batch)
        for t, c in enumerate(batch):
            x = onehot(c)

            Whx = np.matmul(self.Wh, x)
            Uhht1 = np.matmul(self.Uh, self.h[-1])
            ht = np.tanh(Whx + Uhht1 + self.bh)
            Wyht = np.matmul(self.Wy, ht)
            y = softmax(Wyht + self.by)

            self.xs.append(x)
            self.h.append(ht)
            ys[t] = y
            
        return ys

    def backward(self, yOuts, yLabels):
        dWh = np.zeros_like(self.Wh)
        dUh = np.zeros_like(self.Uh)
        dbh = np.zeros_like(self.bh)
        dWy = np.zeros_like(self.Wy)
        dby =  np.zeros_like(self.by)
        dhnext = np.zeros_like(self.h[0])
        for t in reversed(range(len(yLabels))):
            # len(h) == len(yLabels) + 1 so handle the offset
            ht = self.h[t+1]
            htp = self.h[t]

            yLabel = onehot(yLabels[t])                # v x 1
            dy = yOuts[t] - yLabel                     # v x 1
            dby += dy                                  # v x 1
            dWy += np.matmul(dy, ht.T)                 # v x d
            dh = np.matmul(self.Wy.T, dy) + dhnext     # d x 1
            dhraw = (1 - ht * ht) * dh                 # d x 1
            dbh += dhraw
            dUh += np.matmul(dhraw, htp.T)             # d x d
            dWh += np.matmul(dhraw, self.xs[t].T)      # d x v
            dhnext = np.matmul(self.Uh, dhraw)         # d x 1

        for dparam in [dWh, dUh, dWy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWh + L2(self.Wh), dUh + L2(self.Uh), dbh + L2(self.bh), dWy + L2(self.Wy), dby + L2(self.by)

    def loss(self, yOuts, yLabels):
        loss = 0
        for t, c in enumerate(yLabels):
            yLabel = onehot(c)
            loss += ((yOuts[t] - yLabel)**2).sum()
        return loss

    def step(self, learning_rate, dWh, dUh, dbh, dWy, dby):
        for w, dw, m in zip([self.Wh, self.Uh, self.bh, self.Wy, self.by],
                            [dWh, dUh, dbh, dWy, dby],
                            [self.mWh, self.mUh, self.mbh, self.mWy, self.mby]):
            m += dw * dw
            w -= learning_rate * dw / (np.sqrt(m + 1e-8))

rnn = Model()

smoothing = 0.005
loss = (1-1/v)**2 + (1/v)**2
acc = 1/v
for epoch in range(50):
    
    rnn.resetInternal(full=True)
    for count, i in enumerate(range(0, len(data), seq_len)):
        x = data[i:i+seq_len]
        y = data[i+1:i+seq_len+1]
        x = x[:len(y)]

        yOut = rnn.forward(x)
        l = rnn.loss(yOut, y) / len(y)
        acc = sum([getPredictedChar(yOut[t]) == y[t] for t in range(len(y))])/len(y) * smoothing + (1-smoothing) * acc
        loss = l * smoothing + (1 - smoothing) * loss
        g = rnn.backward(yOut, y)
        gu = np.absolute(np.concatenate([gw.flatten() for gw in g])).mean()
        gstd = np.std(np.concatenate([gw.flatten() for gw in g]))
        if count % 100 == 0:
            print(f'epoch {epoch+i/len(data):6.2f} loss {loss:10.4f} {acc*100:6.2f}%', end='\r')
        rnn.step(1e-1, *g)
        rnn.resetInternal()

    print(f'epoch {epoch+1:6} loss {loss:10.4f} {acc*100:6.2f}%')

rnn.resetInternal(full=True)
seed = ['<VERSE_NUM>']
yOut = rnn.forward(seed)
print(''.join([token2printable(t) for t in seed]), end=' ')
for _ in range(10000 - len(seed)):
    idx = np.random.choice(len(VOCAB), p=yOut[-1].ravel())
    c = VOCAB[idx]
    print(token2printable(c), end='')
    yOut = rnn.forward([c])
print()

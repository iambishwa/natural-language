import numpy as np
import pandas as pd
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import autograd

DATASET_INPUT_PATH = '../snli_1_tiny/train.json'
EMBEDDING_DIMENSION = 600
HIDDEN_DIM = 200

class Reader ():

    def __init__ (self, inp_path=DATASET_INPUT_PATH):
        self.vocab = []
        self.inp_path = inp_path
        self.word_to_ix = {}
        self.premise = []
        self.hypothesis = []

    '''
    Reads the input dataset and returns the vocabulary (words as tokens)
    '''
    def read (self):
        df = pd.read_json(self.inp_path)

        for i in np.arange(df.shape[0]):
            for x in df['sentence1'][i].split():
                self.vocab.append(x.lower())
            for x in df['sentence2'][i].split():
                self.vocab.append(x.lower())

        self.vocab = set(self.vocab)
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        return df;

    def embed (self):
        df = self.read();
        embeds = nn.Embedding(len(self.vocab), EMBEDDING_DIMENSION);

        for i in np.arange(0, df.shape[0]):
            sentence1 = []
            flag = False
            for x in df['sentence1'][i].split():
                lookup_tensor = torch.tensor([self.word_to_ix[x.lower()]], dtype=torch.long)
                word_embed = embeds(lookup_tensor)
                word_embed = word_embed.data.numpy()
                word_embed = word_embed.reshape(1, EMBEDDING_DIMENSION)

                if flag == False:
                    sentence1 = word_embed
                    flag = True
                else:
                    sentence1 = np.concatenate([sentence1, word_embed], axis=0)
            self.premise.append(sentence1)

            sentence2 = []
            flag = False
            for x in df['sentence2'][i].split():
                lookup_tensor = torch.tensor([self.word_to_ix[x.lower()]], dtype=torch.long)
                word_embed = embeds(lookup_tensor)
                word_embed = word_embed.data.numpy()
                word_embed = word_embed.reshape(1, EMBEDDING_DIMENSION)

                if flag == False:
                    sentence2 = word_embed
                    flag = True
                else:
                    sentence2 = np.concatenate([sentence2, word_embed], axis=0)
            self.hypothesis.append(sentence2)
        
        return self.premise, self.hypothesis


class Model (nn.Module):

    def __init__(self, batch_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(600, HIDDEN_DIM)
        self.linear2 = nn.Linear(200, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, bidirectional=True)
        self.lstm2 = nn.LSTM(HIDDEN_DIM*2, HIDDEN_DIM, bidirectional=True)
        self.hidden1 = autograd.Variable(torch.zeros(2, batch_size, HIDDEN_DIM)),autograd.Variable(torch.zeros(2, batch_size, HIDDEN_DIM))
        self.hidden2 = autograd.Variable(torch.zeros(2, batch_size, HIDDEN_DIM)),autograd.Variable(torch.zeros(2, batch_size, HIDDEN_DIM))

    '''
    This function does the positionWiseFFN
    '''
    def forward (self, input):
        a1 = F.relu(self.linear1(input))
        a2 = F.relu(self.linear2(a1))

        lstmOut1, self.hidden2 = self.lstm(a2, self.hidden1)
        lstmOut1 = lstmOut1.view(-1, 1, HIDDEN_DIM*2)

        lstmOut2, _ = self.lstm2(lstmOut1, self.hidden2)

        return lstmOut2
        

if __name__ == '__main__':
    reader = Reader()
    premise, hypothesis = reader.embed()
    model = Model(1)

    for i in np.arange(0,len(premise)):
        premise[i] = (torch.tensor(premise[i], requires_grad=False)).reshape(-1, 1, 600)
        hypothesis[i] = (torch.tensor(hypothesis[i], requires_grad=False)).reshape(-1, 1, 600)

    print('Forward propagation started...')
    for inp in premise:

        # print('inp.shape: ', inp.shape)
        model(inp)
    print('Forward propagation till contextual encoding layer finished')









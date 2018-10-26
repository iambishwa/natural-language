import numpy as np
import pandas as pd
import torch
from torch import nn as nn
import torch.nn.functional as F

DATASET_INPUT_PATH = '../snli_1_tiny/train.json'
EMBEDDING_DIMENSION = 600

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

        print('df: ', df);

        for i in np.arange(0, df.shape[0]):
            sentence1 = []
            for x in df['sentence1'][i].split():
                lookup_tensor = torch.tensor([self.word_to_ix[x.lower()]], dtype=torch.long)
                word_embed = embeds(lookup_tensor)
                sentence1.append(word_embed.data.numpy())
            self.premise.append(sentence1)

            sentence2 = []
            for x in df['sentence2'][i].split():
                lookup_tensor = torch.tensor([self.word_to_ix[x.lower()]], dtype=torch.long)
                word_embed = embeds(lookup_tensor)
                sentence2.append(word_embed.data.numpy())
            self.hypothesis.append(sentence2)
            return self.premise, self.hypothesis


class Model (nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(600, 200)
        self.linear2 = nn.Linear(200, 200)

    '''
    This function does the positionWiseFFN
    '''
    def forward (self, input):
        a1 = F.relu(self.linear1(input))
        a2 = F.relu(self.linear2(a1))

        print('a2.shape: ', a2.shape)
        return a2
        

if __name__ == '__main__':
    reader = Reader()
    premise, hypothesis = reader.embed()
    model = Model()

    premise = torch.tensor(premise, requires_grad=False)

    for inp in premise:

        model(premise)









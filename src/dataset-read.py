import numpy as np
import pandas as pd

DATASET_INPUT_PATH = '../snli_1_tiny/train.json'

class Reader:

    def __init__ (self, inp_path=DATASET_INPUT_PATH):
        self.vocab = []
        self.inp_path = inp_path

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
        return self.vocab

if __name__ == '__main__':
    reader = Reader()
    print(reader.read())

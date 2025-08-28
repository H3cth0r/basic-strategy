import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import math
import random
import os
import sys
from typing import List, Dict, Tuple

from datasets import load_dataset


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer 
# Neural network cannot understand raw text. It only understands
# numbers. A tokenizer's job is to convert text into a sequence of numbers
# called tokenization or encoding. Then it will convert sequence back into 
# human readable text (decoding). This tokenization works at the character
# lelve, meaning each character gets a unique number.
class CharTokenizer:
    def __init__(self, texts: List[str], lower: bool=True, min_freq: int=1):
        """
        texts (List[str]): list of sequences to build the vocabulary from.
        lower (bool): if true, all text is converted to lowercase
        min_freq (int): minimum number of times a character must appear to be included in the vocab
        """
        self.lower = lower
        
        # these tokens have special meanings and are essential for the model.
        self.PAD = "<pad>"      # Padding Token: Used to make all sequences in a batch the same length.
        self.SOS = "<s>"        # Start-of-Sequence: Marks the beginning of a sentence.
        self.EOS = "</s>"       # End-of-Sequence: Marks the end of a sentence.
        self.UNK = "<unk>"      # Unknown Token: Represents any character not in our vocabulary.

        # --- Building the Vocabulary ---
        # count character frequencies.
        freq = {}
        for t in texts:
            if t is None: continue
            s = t.lower() if lower else t
            for ch in s:
                freq[ch] = freq.get(ch, 0) + 1

        # create itos (integer-to-string) mapping. This is a list where the
        #    index is the ID and the value is the character.
        self.itos = [self.PAD, self.SOS, self.EOS, self.UNK]
        # sort characters by frequency (most common first) to build a robust vocabulary.
        chars = [c for c, f in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if f >= min_freq]
        self.itos += [c for c in chars if c not in self.itos]

        # create stoi (string-to-integer) mapping. This is a dictionary for
        #    fast lookups of a character's ID.
        self.stoi = {s: i for i, s in enumerate(self.itos)}

        # we store the integer IDs of special tokens for easy access later. For example,
        # when we need to pad a batch, we can directly use self.pad_id.
        self.pad_id = self.stoi[self.PAD] # The ID for the padding token.
        self.sos_id = self.stoi[self.SOS] # The ID for the start-of-sequence token.
        self.eos_id = self.stoi[self.EOS] # The ID for the end-of-sequence token.
        self.unk_id = self.stoi[self.UNK] # The ID for the unknown token.

def main():

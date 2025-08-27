import torch
import torch.nn as nn
import torch.optim as optim

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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Responsible for converting text into a sequence of numbers (integers) and back
# Operates at the character level, meaning each character becomes a unique token
class CharTokenizer:
    """
    Manages the vocabulary and conversion between cahracters and Integer IDs.
    """
    def __init__(self, texts: List[str], lower: bool=True, min_freq: int=1):
        """
        Initializes the tokenizer and builds the vocabulary.
        Args:
            texts: list of sentences to build the vocabulary from.
            lower(bool): if true, all text will be converted to lowercase.
            min_freq: minimum frequency a character must have to be included in the vocabulary.
        """
        # Definition of tokens that have specific meanings
        self.lower  = lower
        self.PAD    = "<pad>"   # Padding token: used to make all sequences in a batch the same length
        self.SOS    = "<s>"     # Start of sentence token: marks the beginning of a sequence
        self.EOS    = "</s>"    # end of sentence token: marks the end of a sequence
        self.UNK    = "<unk>"   # represents any character not in vocab
        
        # build vocab from characters (no whitespace splitting)
        freq = {}
        for t in texts:
            if t is None: continue
            s = t.lower() if lower else t
            for ch in s:
                freq[ch] = freq.get(ch, 0) + 1

        print(freq)
        # Special tokens first
        self.itos = [self.PAD, self.SOS, self.EOS, self.UNK]
        # Add chars by frequency
        chars = [c for c, f in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if f >= min_freq]
        print(chars)
        # for c in chars:
        #     if c not in self.itos:
        #         self.itos.append(c)
        self.itos += chars
        print(self.itos)

        self.stoi = {s: i for i, s in enumerate(self.itos)}
        print(self.stoi)
        self.pad_id = self.stoi[self.PAD]
        self.sos_id = self.stoi[self.SOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]
        print(self.pad_id)

class TranslationPairDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            pairs: List[Tuple[str, str]],
            src_tok: CharTokenizer,
            tgt_tok: CharTokenizer,
            max_src_len: int = 140,
            max_tgt_len: int = 140,
    ):
        self.data       = pairs
        self.src_tok    = src_tok
        self.tgt_tok    = tgt_tok
        self.max_src_len=max_src_len
        self.max_tgt_len=max_tgt_len

def collate_batch(batch, pad_src: int, pad_tgt: int):
    src_seqs, tgt_seqs = zip(*batch)

def main():
    ds = load_dataset("bentrevett/multi30k")
    df_train        = ds["train"].to_pandas()
    df_validation   = ds["validation"].to_pandas()

    src_texts = df_train["en"]
    tgt_texts = df_train["de"]

    print(src_texts)
    print(tgt_texts)

    src_tok = CharTokenizer(src_texts, lower=True, min_freq=1)
    tgt_tok = CharTokenizer(tgt_texts, lower=True, min_freq=1)

    max_src_len = 140
    max_tgt_len = 140
    train_ds    = TranslationPairDataset(df_train, src_tok, tgt_tok, max_src_len, max_tgt_len)
    valid_ds    = TranslationPairDataset(df_validation, src_tok, tgt_tok, max_src_len, max_tgt_len)

if __name__ == "__main__":
    main()

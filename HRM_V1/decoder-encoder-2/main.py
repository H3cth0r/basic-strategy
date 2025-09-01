import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
from datasets import load_dataset

# import torchtext
import tqdm
# import evaluate


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

if  __name__ == "__main__":
    # Load Dataset
    dataset = load_dataset("bentrevett/multi30k")
    print(dataset)
    print("="*20)

    # Separate dataset into specific vars
    train_data, valid_data, test_data = (
            dataset["train"],
            dataset["validation"],
            dataset["test"],
    )
    print(train_data[0])
    print("="*20)

    # Load tokenizers
    en_nlp = spacy.load("en_core_web_sm")
    print(en_nlp._path)
    de_nlp = spacy.load("de_core_news_sm")
    print("="*20)

    # Test Tokenization
    string = "What a lovely day it is today!"
    print([token.text for token in en_nlp.tokenizer(string)])
    print("="*20)

    # apply and add tokens
    max_length = 1_000
    lower = True
    sos_token = "<sos>"
    eos_token = "<eos>"
    fn_kwargs = { 
            "en_nlp": en_nlp,
            "de_nlp": de_nlp,
            "max_length": max_length,
            "lower": lower,
            "sos_token": sos_token,
            "eos_token": eos_token,
    }
    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

    print(train_data)
    print(train_data[0])
    print("="*20)

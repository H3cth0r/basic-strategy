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
    
    def encode(self, text: str, add_specials: bool=True, max_len: int=None) -> List[int]:
        """ Converts a string of text into a list of integer IDs."""
        if text is None: text = ""
        s = text.lower() if self.lower else text
        ids = []
        if add_specials:
            ids.append(self.sos_id) # Start with the Start-of-Sequence token
        for ch in s:
            ids.append(self.stoi.get(ch, self.unk_id)) # use UNK id if character is not in vocab
            # truncate if the sequence exceeds max_len
            if max_len is not None and len(ids) >= (max_len - 1 if add_specials else max_len):
                break
        if add_specials:
            ids.append(self.eos_id) # End with the End-of-Sequence token
        return ids
    def decode(self, ids: List[int], remove_specials: bool=True) -> str:
        """ Converts a list of integer IDs back into a string of text """
        chars = []
        for i in ids:
            if remove_specials and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            # Look up the character for the ID. Default to UNK if ID is invalid
            token = self.itos[i] if 0 <= i < len(self.itos) else self.UNK
            if not remove_specials or token not in (self.PAD, self.SOS, self.EOS, self.UNK):
                chars.append(token)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        """ Returns the total number of unique tokens in the vocabulary """
        return len(self.itos)

class TranslationPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_tok: CharTokenizer,
        tgt_tok: CharTokenizer,
        max_src_len: int = 140,
        max_tgt_len: int = 140,
    ):
        self.data           = pairs
        self.src_tok        = src_tok
        self.tgt_tok        = tgt_tok
        self.max_src_len    = max_src_len
        self.max_tgt_len    = max_tgt_len

    def __len__(self):
        """ Returns the total number of sentence pairs """
        return len(self.data)
    def __getitem__(self, idx):
        """ Retrieves and toenizes the source and target sentences at a given index """
        src_text, tgt_text = self.data[idx]
        # Use the tokenizers encode method to convert text to a list of ids
        src_ids = self.src_tok.encode(src_text, add_specials=True, max_len=self.max_src_len)
        tgt_ids = self.tgt_tok.encode(tgt_text, add_specials=True, max_len=self.max_tgt_len)
        # return them as pytorch tensors
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# Collate Function
# When creating a batch of data, the sentences will have different lengths. However,
# tensors in PyTorch must have a uniform shape. The collate_fn solves this by taking a list
# of samples and padding them. Padding means adding special <pad> tokens to the end of shorter
# sentences until all sequences in the batch are the same length as the longest one.
def collate_batch(batch, pad_src: int, pad_tgt: int):
    """pads sequences in a batch to the same length"""
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)

    # Create empty tensors filled with the padding ID
    max_src     = max(len(s) for s in src_seqs)
    max_tgt     = max(len(s) for t in tgt_seqs)
    padded_src = torch.full((len(batch), max_src), pad_src, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), pad_tgt, dtype=torch.long)

    # Copy the actual sequence data into the padded tensors
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        padded_src[i, :len(s)] = s
        padded_tgt[i, :len(t)] = t

    # Prepare Decoder Inputs and Targets
    # Decoder needs two versions of the target sequence
    # 1. tgt_in: the input to the decoder (e.g., "<s> hello world")
    # 2. tgt_out: the target the decoder should predict (e.g., "hello world </s>")
    # This is how the model learns to predict the next token in the sequence
    tgt_in = padded_tgt[:, :-1].contiguous()
    tgt_out = padded_tgt[:, 1:].contiguous()

    return padded_src, src_lens, tgt_in, tgt_out

# The Encoder-Decoder Model
# Architecture designed for sequence-to-sequence tasks like translation
# - The Encoder: Reads the entire input sequence ("a man is happy") and
#   compresses all its information into a single vector representation, often called 
#   the "context vector" or "hidden state".

# - The Decoder: Takes the encoder's context vector and generates the output sequence
#   one token at a time (e.g., "e", "i", "n", " ", "m", ...)
class Encoder(nn.Module):
    """The Encoder part of the Seq2Sec model."""
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Embedding Layer: Converts integer IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # GRU (Gated Recurrent Unit): A type of RNN that processes the sequence
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        # outputs contains the hidden state from every timestep
        # hidden is the final hidden state of the sequence (the context vector)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden.squeeze(0)

# Attention
# Without attention, the decoder has to rely solely on the siingle context vector from
# the encoder. For long sentences, this is a bottleneck. The attention mechanism allows
# The attention mechanism allows the decoder, at every step of generating an output, to
# "look abck" at the entire input sequence and focus on the most relevant parts. For
# example, when translating "man" to "mann", the attention should be high on the word
# "man" in the source.
class DotAttention(nn.Module):
    """ A simple Dot-Product Attention mechanism """
    def forward(self, query, keys, mask):
        # =====================================================================
        # Step 0: The Inputs
        # =====================================================================
        # query: The decoder's current hidden state. It's the "question" about
        #        what to focus on in the source.
        #        Shape: [Batch_Size, Hidden_Dim] -> [1, 256]
        #
        # keys: All the output hidden states from the Encoder. This is the
        #       "source document" or information we can pay attention to.
        #       Shape: [Batch_Size, Src_Len, Hidden_Dim] -> [1, 5, 256]
        #
        # mask: A tensor indicating which parts of the 'keys' are real data
        #       and which are just padding. 1 for real, 0 for padding.
        #       Example: [1, 1, 1, 1, 0] if the source sentence has 4 real tokens
        #       and 1 padding token.
        #       Shape: [Batch_Size, Src_Len] -> [1, 5]
        # =====================================================================

        # =====================================================================
        # Step 1: Calculate Scores (Measuring Relevance)
        # =====================================================================
        # CONCEPT: What are scores? Scores measure the similarity or "relevance"
        # between the decoder's current thought (query) and every single token
        # from the source sentence (keys). We use the dot product for this, which
        # is a simple and effective way to measure how well two vectors align.
        #
        # torch.bmm performs a Batch Matrix Multiplication. We need to make the
        # shapes compatible:
        #   keys:          [1, 5, 256]
        #   query.unsqueeze(2): [1, 256, 1]  (We add a dimension to make it a matrix)
        #
        # The result of `bmm` will have shape [1, 5, 1].
        # .squeeze(2) removes the last dimension.
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        #
        # DATA AT THIS STEP (`scores`): A vector of raw numbers. A higher number
        # means the corresponding source token is more relevant to the current query.
        # Shape: [Batch_Size, Src_Len] -> [1, 5]
        # Example Value: tensor([[10.1, 2.3, 15.8, 8.2, 9.1]])
        # This means the 3rd token (index 2) is the most relevant right now.
        # =====================================================================

        # =====================================================================
        # Step 2: Masking (Ignoring Padding)
        # =====================================================================
        # We don't want the model to pay attention to `<pad>` tokens. We set the
        # score for any padded position to a very large negative number.
        scores = scores.masked_fill(mask == 0, -1e9)
        #
        # DATA AT THIS STEP (`scores` with mask=[1,1,1,1,0]):
        # Example Value: tensor([[10.1, 2.3, 15.8, 8.2, -1e9]])
        # The score for the padded token is now effectively negative infinity.
        # =====================================================================

        # =====================================================================
        # Step 3: Getting Probabilities (Softmax)
        # =====================================================================
        # The softmax function converts our raw scores into a probability
        # distribution. The numbers will now be between 0 and 1, and they will
        # all sum to 1. This is our "attention distribution".
        attn = torch.softmax(scores, dim=1)
        #
        # DATA AT THIS STEP (`attn`): A vector of weights.
        # Shape: [Batch_Size, Src_Len] -> [1, 5]
        # Example Value: tensor([[0.25, 0.01, 0.60, 0.14, 0.00]])
        # This tells the decoder to focus 60% on the 3rd token, 25% on the 1st,
        # 14% on the 4th, etc., and 0% on the padded token.
        # =====================================================================

        # =====================================================================
        # Step 4: Creating the Context Vector (Weighted Sum)
        # =====================================================================
        # CONCEPT: Now we use the attention weights (`attn`) to create a
        # "weighted average" of the encoder outputs (`keys`). This creates a
        # single vector that blends the information from the source sentence,
        # emphasizing the parts we decided were most relevant.
        #
        # attn.unsqueeze(1): [1, 1, 5]
        # keys:              [1, 5, 256]
        # `bmm` result:      [1, 1, 256]
        # .squeeze(1) removes the middle dimension.
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)
        #
        # DATA AT THIS STEP (`context`): The final context vector.
        # Shape: [Batch_Size, Hidden_Dim] -> [1, 256]
        # This vector is a rich summary of the source sentence, specifically
        # tailored for generating the *current* target token.
        # =====================================================================

        return context, attn

class Decoder(nn.Module):
    """ The Decoder part of the Seq2Seq model"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = DotAttention()
        # The RNN input is the concatenation of the current word embedding and the attention context
        self.rnn = nn.GRU()
        # A final linear layer to project the output to the size of the vocabulary
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, x_t, hidden, enc_outputs, src_mask):
        """ Performs a single decoding step. """
        # =====================================================================
        # Step 0: The Inputs for this Timestep
        # =====================================================================
        # x_t: The input token for this specific step. At the first step, this is
        #      the `<s>` (SOS) token. At later steps, it's the token generated
        #      in the previous step.
        #      Shape: [Batch_Size] -> [1] (containing a single token ID)
        #
        # hidden: The decoder's RNN hidden state from the PREVIOUS step.
        #         This is our "memory" of what we've translated so far.
        #         Shape: [Batch_Size, Hidden_Dim] -> [1, 256]
        #
        # enc_outputs: ALL hidden states from the encoder. This never changes
        #              during the decoding of a single sentence.
        #              Shape: [Batch_Size, Src_Len, Hidden_Dim] -> [1, 5, 256]
        #
        # src_mask: The padding mask for the source sentence.
        #           Shape: [Batch_Size, Src_Len] -> [1, 5]
        # =====================================================================

        # =====================================================================
        # Step 1: Embed the Input Token
        # =====================================================================
        # Convert the token ID into a dense vector representation.
        # Note: self.embedding expects a tensor of shape (Batch, SeqLen), so
        # we unsqueeze to add a temporary sequence length of 1, which we later remove.
        # The provided code simplifies this, but under the hood, this is what happens.
        # For simplicity, we'll assume the embedding layer handles a [Batch_Size] input.
        emb = self.dropout(self.embedding(x_t))
        #
        # DATA AT THIS STEP (`emb`): The vector for the current input character.
        # Shape: [Batch_Size, Emb_Dim] -> [1, 128] (assuming emb_dim=128)
        # =====================================================================

        # =====================================================================
        # Step 2: GET THE ATTENTION CONTEXT (The Magic Happens Here)
        # =====================================================================
        # We call our attention module. Notice the mapping:
        #   - The `query` is the decoder's previous `hidden` state.
        #   - The `keys` are the `enc_outputs`.
        #   - The `mask` is the `src_mask`.
        context, attn = self.attn(hidden, enc_outputs, src_mask)
        #
        # DATA AT THIS STEP:
        # `context`: A summary of the source sentence, tailored to this step.
        #            Shape: [Batch_Size, Hidden_Dim] -> [1, 256]
        # `attn`: The attention weights for visualization.
        #         Shape: [Batch_Size, Src_Len] -> [1, 5]
        # =====================================================================

        # =====================================================================
        # Step 3: Combine and Feed to the RNN
        # =====================================================================
        # We provide the RNN with two pieces of information:
        # 1. The current character we're looking at (`emb`).
        # 2. The relevant context from the source sentence (`context`).
        # We concatenate them into a single, richer input vector.
        rnn_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
        #
        # DATA AT THIS STEP (`rnn_in`): The combined vector.
        # Shape: [Batch_Size, Seq_Len, Emb_Dim + Hidden_Dim] -> [1, 1, 128 + 256]
        # The `unsqueeze(1)` is to give it a sequence length of 1, as RNNs expect
        # a 3D tensor: (Batch, Seq_Len, Input_Dim).
        # =====================================================================

        # =====================================================================
        # Step 4: Update the Decoder State with the RNN
        # =====================================================================
        # The GRU processes the combined input (`rnn_in`) and the previous
        # hidden state (`hidden`) to produce a new output and a new hidden state.
        out, h_new = self.rnn(rnn_in, hidden.unsqueeze(0))
        # DATA AT THIS STEP:
        # `out`: The output of the GRU at this timestep.
        #        Shape: [Batch_Size, Seq_Len, Hidden_Dim] -> [1, 1, 256]
        # `h_new`: The NEW hidden state for the decoder. This becomes the `hidden`
        #          input for the *next* timestep. This is how memory is passed.
        #          Shape: [Num_Layers, Batch_Size, Hidden_Dim] -> [1, 1, 256]
        # We squeeze them to remove the sequence/layer dimensions.
        # =====================================================================

        # =====================================================================
        # Step 5: Make a Prediction
        # =====================================================================
        # We concatenate the RNN output (`out`) and the attention context (`context`)
        # one last time before feeding it to the final linear layer (`fc_out`).
        # This gives the final layer maximum information to make a good prediction.
        pred_input = torch.cat([out.squeeze(1), context], dim=-1)
        # Shape of `pred_input`: [Batch_Size, Hidden_Dim + Hidden_Dim] -> [1, 512]
        
        logits = self.fc_out(pred_input)
        #
        # DATA AT THIS STEP (`logits`): Raw scores for every character in the
        # target vocabulary. The character with the highest score is our prediction.
        # Shape: [Batch_Size, Vocab_Size] -> [1, 58] (if vocab has 58 chars)
        # =====================================================================

        return logits, h_new.squeeze(0), attn

class Seq2Seq(nn.Module):
    """ Main model that combines the Encoder and Decoder."""
    def __init__(self, enc: Encoder, dec: Decoder, src_pad_id: int):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.src_pad_id = src_pad_id

    def make_src_mask(self,  src):
        """Creates a mask to identify padding tokens in the soruce. """
        # =====================================================================
        # Step 2: Creating the Attention Mask
        # =====================================================================
        # Input `src`: A tensor of source token IDs.
        # Shape: [Batch_Size, Src_Len]
        #
        # Logic: This performs a simple element-wise comparison. For every
        # token ID in the `src` tensor, it checks: "Is this ID NOT the padding ID?".
        # The result is a boolean tensor (True/False).
        # .long() converts True to 1 and False to 0.
        #
        # Output: A tensor where 1s represent real tokens and 0s represent
        # padding tokens. This mask is fed to the DotAttention module to tell
        # it "you are not allowed to pay attention to the 0s".
        # Shape: [Batch_Size, Src_Len]
        return (src != self.src_pad_id).long()

    def forward(self, src, src_lens, tgt_in, teacher_forcing=True):
        # =====================================================================
        # Step 1: Encoding the Source
        # =====================================================================
        # This line runs the entire batch of source sentences through the Encoder.
        enc_out, enc_h = self.enc(src, src_lens)
        #
        # DATA AT THIS STEP:
        # `enc_out`: The output hidden state from *every* timestep of the encoder.
        #            This is the "source document" the decoder will look at.
        #            Shape: [B, Src_Len, H] -> [2, 5, 256]
        # `enc_h`: The *final* hidden state from the encoder. This is the "thought
        #          vector" or context that summarizes the entire source sentence.
        #          Shape: [B, H] -> [2, 256]
        # =====================================================================
        enc_out, enc_h = self.enc(src, src_lens)
        src_mask = self.make_src_mask(src)

        B, T = tgt_in.shape
        logits_list = []
        dec_h = enc_h
        x_t = tgt_in[:, 0]

        for t in range(T):
            step_logits, dec_h, _ = self.dec.forward_step(x_t, dec_h, enc_out, src_mask)
            logits_list.append(step_logits.unsqueeze(1))

            # What is teacher forcing
            # During training, we can either feed the decoders own previous prediction
            # as the next input, or we can ffed the actual correct toekn from the
            # target sequence. The latter is called teacher forcing. It stabilizes
            # training and helps the model learn faster
            if teacher_forcing:
                x_t = tgt_in[:, t+1] if t < T - 1 else x_t
            else:
                x_t = step_logits.argmax(-1)

        return torch.cat(logits_list, dim=1)

    def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
        """ Geneartes a translation during inference using a greedy approach"""
        self.eval()
        with torch.no_grad():
            enc_out, enc_h = self.enc(src, src_lens)
            src_mask = self.make_src_mask(src)

            B = src.size(0)
            generated   = [[] for _ in range(B)]
            attn_all    = [[] for _ in range(B)]
            x_t = torch.full((B,), sos_id, dtype=torch.long, device=src.device)
            dec_h = enc_h

            for _ in range(max_len):
                step_logits, dec_h, attn = self.dec.forward_step(x_t, dec_h, enc_out, src_mask)
                x_t = step_logits.argmax(dim=1)
                for i in range(B):
                    if x_t[i].item() == eos_id: continue
                    generated[i].append(x_t[i].item())
                    attn_all[i].append(attn[i].detach().cpu())
            return generated, attn_all

def main():
    ds = load_dataset("bentrevett/multi30k")
    df_train = ds["train"].to_pandas()
    df_valid = ds["validation"].to_pandas()

    # Create pairs of (source, target) sentences. This list of tuples is the
    # format expected by our custom TranslationPairDataset
    train_pairs = list(zip(df_train['en'], df_train['de']))
    valid_pairs = list(zip(df_valid['en'], df_valid['de']))

    # Build tokenizers
    src_texts = df_train['en'].tolist()
    tgt_texts = df_train['de'].tolist()
    src_tok = CharTokenizer(src_texts, lower=True, min_freq=1)
    tgt_tok = CharTokenizer(tgt_texts, lower=True, min_freq=1)

    # Create datasets and DataLoaders
    max_src_len = 140
    max_tgt_len = 140
    train_ds    = TranslationPairDataset(train_pairs, src_tok, tgt_tok, max_src_len, max_tgt_len)
    valid_ds    = TranslationPairDataset(valid_pairs, src_tok, tgt_tok, max_src_len, max_tgt_len)

    # The collate function needs to be wrapped so we can pass the padding IDs
    def collate_fn(batch):
        return collate_batch(batch, pad_src=src_tok.pad_id, pad_tgt=tgt_tok.pad_id)

    # The DataLoader will automatically use the DAtaset and the collate_fn
    # to create efficient, padded batches for training
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print(train_loader)

    # Initialize Model, Loss and Optimizer
    emb_dim = 128
    hid_dim = 256
    enc = Encoder(src_tok.vocab_size, emb_dim, hid_dim, dropout=0.2)
    enc = Decoder(tgt_tok.vocab_size, emb_dim, hid_dim, dropout=0.2)

if __name__ == "__main__":
    main()

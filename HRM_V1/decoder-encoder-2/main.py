import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
from datasets import load_dataset

from torchtext.vocab import build_vocab_from_iterator
import tqdm
import evaluate


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
def numeralize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
                "en_ids": batch_en_ids,
                "de_ids": batch_de_ids,
        }
        return batch
    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
    )
    return data_loader

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.n_layers       = n_layers 
        self.embedding      = nn.Embedding(input_dim, embedding_dim) 
        self.rnn            = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout        = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim     = output_dim
        self.hidden_dim     = hidden_dim 
        self.n_layers       = n_layers
        self.embedding      = nn.Embedding(output_dim, embedding_dim)
        self.rnn            = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out         = nn.Linear(hidden_dim, output_dim)
        self.dropout        = nn.Dropout(dropout)
    def forward(self, inputt, hidden, cell):
        inputt                  = inputt.unsqueeze(0)
        embedded                = self.dropout(self.embedding(inputt))
        output, (hidden, cell)  = self.rnn(embedded, (hidden, cell))
        prediction              = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.device = device
        assert (
                encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal"
        assert (
                encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers"

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size      = trg.shape[1]
        trg_length      = trg.shape[0]
        trg_vocab_size  = self.decoder.output_dim
        outputs         = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        hidden, cell    = self.encoder(src)
        inputt          = trg[0, :]
        for t in range(1, trg_length):
            output, hidden, cell    = self.decoder(inputt, hidden, cell)
            outputs[t]              = output
            teacher_force           = random.random() < teacher_forcing_ratio
            top1                    = output.argmax(1)
            inputt                   = trg[t] if teacher_force else top1
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_fn(
        model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src         = batch["de_ids"].to(device)
        trg         = batch["en_ids"].to(device)
        optimizer.zero_grad()
        output      = model(src, trg, teacher_forcing_ratio)
        output_dim  = output.shape[-1]
        output      = output[1:].view(-1, output_dim)
        trg         = trg[1:].view(-1)
        loss        = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens

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

    # Build Vocab
    min_freq = 2
    unk_token = "<unk>"
    pad_token = "<pad>"
    special_tokens = [
            unk_token,
            pad_token,
            sos_token,
            eos_token,
    ]
    en_vocab = build_vocab_from_iterator(
            train_data["en_tokens"],
            min_freq=min_freq,
            specials=special_tokens
    )
    de_vocab = build_vocab_from_iterator(
            train_data["de_tokens"],
            min_freq=min_freq,
            specials=special_tokens
    )
    # itos: int to string
    print(en_vocab.get_itos()[:10])
    print(en_vocab.get_itos()[9])
    print(en_vocab.get_stoi()["the"])
    print(de_vocab.get_itos()[:10])
    print("len unique tokens en: ", len(en_vocab))
    print("len unique tokens de: ", len(de_vocab))
    print("="*20)

    # Check index for unkwon index
    assert en_vocab[unk_token] == de_vocab[unk_token]
    assert en_vocab[pad_token] == de_vocab[pad_token]

    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token]
    print(unk_index)
    print("="*20)

    # Manually set the defualt token index when unkwon token
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)
    print("="*20)

    tokens = ["i", "love", "watching", "crime", "shows"]
    print(en_vocab.lookup_indices(tokens))
    print(en_vocab.lookup_tokens(en_vocab.lookup_indices(tokens)))
    print("="*20)

    fn_kwargs = {
            "en_vocab": en_vocab,
            "de_vocab": de_vocab
    }
    train_data  = train_data.map(numeralize_example, fn_kwargs=fn_kwargs)
    valid_data  = valid_data.map(numeralize_example, fn_kwargs=fn_kwargs)
    test_data   = test_data.map(numeralize_example, fn_kwargs=fn_kwargs)
    print(train_data)
    print(train_data[0])
    print("="*20)


    data_type = "torch"
    format_columns = ["en_ids", "de_ids"]
    train_data = train_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True
    )
    valid_data = valid_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True
    )
    test_data = test_data.with_format(
            type=data_type, columns=format_columns, output_all_columns=True
    )
    print(train_data[0])
    print("="*20)

    # Collator
    batch_size = 128
    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)
    print(train_data_loader)
    print("="*20)


    # Init Model
    input_dim               = len(de_vocab)
    output_dim              = len(en_vocab)
    encoder_embedding_dim   = 256
    decoder_embedding_dim   = 256
    hidden_dim              = 512
    n_layers                = 2
    encoder_dropout         = 0.5
    decoder_dropout         = 0.5
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder = Encoder(
            input_dim,
            encoder_embedding_dim,
            hidden_dim,
            n_layers,
            encoder_dropout,
    )
    decoder = Decoder(
            output_dim,
            decoder_embedding_dim,
            hidden_dim,
            n_layers,
            decoder_dropout,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    model.apply(init_weights)
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")
    print("="*20)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    print("="*20)

    n_epochs = 10
    clip = 1.0
    teacher_forcing_ratio = 0.5
    best_valid_loss = float("inf")
    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss = train_fn(
            model,
            train_data_loader,
            optimizer,
            criterion,
            clip,
            teacher_forcing_ratio,
            device,
        )
        valid_loss = evaluate_fn(
            model,
            valid_data_loader,
            criterion,
            device,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
    print("="*20)

    model.load_state_dict(torch.load("tut1-model.pt"))
    test_loss = evaluate_fn(model, test_data_loader, criterion, device)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")
    print("="*20)


    sentence = test_data[0]["de"]
    expected_translation = test_data[0]["en"]
    translation = translate_sentence(
        sentence,
        model,
        en_nlp,
        de_nlp,
        en_vocab,
        de_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )
    print("Input: ", sentence)
    print("Expected: ", expected_translation)
    print("Translation: ", translation)
    print("="*20)

    sentence = "Ein Mann sitzt auf einer Bank."
    translation = translate_sentence(
        sentence,
        model,
        en_nlp,
        de_nlp,
        en_vocab,
        de_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )
    print("Input: ", sentence)
    print("Translation: ", translation)
    print("="*20)

    translations = [
        translate_sentence(
            example["de"],
            model,
            en_nlp,
            de_nlp,
            en_vocab,
            de_vocab,
            lower,
            sos_token,
            eos_token,
            device,
        )
        for example in tqdm.tqdm(test_data)
    ]
    bleu = evaluate.load("bleu")
    predictions = [" ".join(translation[1:-1]) for translation in translations]
    references = [[example["en"]] for example in test_data]
    print("Input: ", predictions[0])
    print("Expected: ", references[0])

    def get_tokenizer_fn(nlp, lower):
        def tokenizer_fn(s):
            tokens = [token.text for token in nlp.tokenizer(s)]
            if lower:
                tokens = [token.lower() for token in tokens]
            return tokens

        return tokenizer_fn

    tokenizer_fn = get_tokenizer_fn(en_nlp, lower)
    tokenizer_fn(predictions[0]), tokenizer_fn(references[0][0])

    results = bleu.compute(
        predictions=predictions, references=references, tokenizer=tokenizer_fn
    )
    print(results)

import torch
from datasets import load_dataset
from model_tokenize import get_or_build_tokenizer
from model import build_transformer
from dataset import LangDataset


from torch.utils.data import random_split, DataLoader


def get_datasets(config):
    src_lang = config['src_lang']
    out_lang = config['tgt_lang']
    seq_len = config['seq_len']
    bs = config['batch_size']


    data = load_dataset('opus_books', f'{src_lang}-{out_lang}', split='train')

    # tokenizer
    src_tokenizer = get_or_build_tokenizer(config, data, src_lang)
    out_tokenizer = get_or_build_tokenizer(config, data, out_lang)

    #split the data (90% for training 10% for validation)
    train_size = int(0.9 * len(data))
    val_size = len(data)- train_size

    train_data, val_data = random_split(data, [train_size, val_size])

    # create the datasets
    train_ds = LangDataset(train_data, src_tokenizer, out_tokenizer, src_lang, out_lang, seq_len)
    val_ds = LangDataset(val_data, src_tokenizer, out_tokenizer, src_lang, out_lang, seq_len)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dl, val_dl, src_tokenizer, out_tokenizer

def get_model(config, src_vocab_size, out_vocab_size):
    seq_len = config['seq_len']
    model = build_transformer(src_vocab_size, out_vocab_size,seq_len, seq_len)
    return model




    
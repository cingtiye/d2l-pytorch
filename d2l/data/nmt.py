#import urllib2
import zipfile
import torch
import requests
from io import BytesIO
from torch.utils import data
from .base import Vocab

__all__ = ['load_data_nmt','build_vocab','pad','build_array']

def build_vocab(tokens):
    tokens = [token for line in tokens for token in line]
    return Vocab(tokens, min_freq=3, use_special_tokens=True)

def pad(line, max_len, padding_token):
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))

def build_array(lines, vocab, max_len, is_source):
    lines = [vocab[line] for line in lines]
    if not is_source:
        lines = [[vocab.bos] + line + [vocab.eos] for line in lines]
    array = torch.tensor([pad(line, max_len, vocab.pad) for line in lines])
    valid_len = (array != vocab.pad).sum(1) #第一个维度
    return array, valid_len

def load_data_nmt(batch_size, max_len, num_examples=1e3):
    with open('../data/fra.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    def preprocess_raw(text):
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        out = ''
        for i, char in enumerate(text.lower()):
            if char in (',', '!', '.') and i > 0 and text[i - 1] != ' ':
                out += ' '
            out += char
        return out

    text = preprocess_raw(raw_text)
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))

    src_vocab, tgt_vocab = build_vocab(source), build_vocab(target)
    src_array, src_valid_len = build_array(source, src_vocab, max_len, True)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, max_len, False)
    train_data = data.TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    return src_vocab, tgt_vocab, train_iter

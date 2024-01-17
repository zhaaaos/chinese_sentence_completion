##################################################
# Import the libraries
# model code adapted from : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
# and https://github.com/lyeoni/gpt-pytorch
##################################################
import numpy as np
import random
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim, autograd

##################################################
# The constants
# 'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
# 'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
# 'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
# 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
# 'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
# 'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
# 'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
# 'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
##################################################
FIRST_TIME_RUNNING = False
COMPUTE_TOKENDS = False
TRAINING_DATA_PATH = "./xiaoshuo.txt"

TOKEN_EMBEDDING_DIMENSION = 210  # The dimention of vector representing a token
SEQUENCE_LENGTH = 95  # the length of input sequence
BATCH_SIZE = 64
FULLY_CONNECTED_LAYER_DIMENSION = 2*TOKEN_EMBEDDING_DIMENSION  # dimention of the fully connected layer in the model
NUMBER_OF_ATTENTION_HEADS = 5  # the number of attention heads in the model
NUMBER_OF_LAYERS = 5  # the number of encoder block in the model
LEARNING_RATE = 2e-04
device = "cuda:1"

##################################################
# The required data
##################################################
tokens = set()
token_to_index_dict = {}
token_embedding = []
num_tokens = 0

### get the input text as string
train_str = ''
with open(TRAINING_DATA_PATH, encoding='utf8') as f:
    train_str = "".join(f.readlines()).replace("  ", " ").replace("\n\n", "\n")
print("training string length:", len(train_str))

if COMPUTE_TOKENDS:
    ### find all unique characters in the string
    for charr in train_str:
        if charr not in tokens:
            tokens.add(charr)
    tokens = list(tokens)

    # save the characters array
    with open('tokens_gpt.pkl', 'wb') as file:
        pickle.dump(tokens, file)

    ### get the random token embedding and
    ### token to index in its array disctionary
    token_to_index_dict = {}
    for i in range(len(tokens)):
        token_to_index_dict[tokens[i]] = i

    with open('token_to_index_dict_gtp.pkl', 'wb') as file:
        pickle.dump(token_to_index_dict, file)

else:
    temp_file = open("tokens_gpt.pkl", 'rb')
    tokens = pickle.load(temp_file)
    temp_file.close()

    temp_file = open("token_to_index_dict_gtp.pkl", 'rb')
    token_to_index_dict = pickle.load(temp_file)
    temp_file.close()

num_tokens = len(tokens)
print("Num of of tokens", str(len(tokens)))

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        #         self.decoder = nn.Linear(d_model, ntoken)
        self.o = nn.Linear(self.d_model * SEQUENCE_LENGTH, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    #         self.decoder.bias.data.zero_()
    #         self.decoder.weight.data.uniform_(-initrange, initrange)

    #     def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #         output = self.transformer_encoder(src, src_mask)
        output = self.transformer_encoder(src)
        #         output = self.decoder(output)
        output = self.o(output.view(-1, self.d_model * SEQUENCE_LENGTH))
        return output


transfencoder = ""
opt = ""
if FIRST_TIME_RUNNING:

    transfencoder = TransformerModel(ntoken=num_tokens,
                                     d_model=TOKEN_EMBEDDING_DIMENSION,
                                     nhead=NUMBER_OF_ATTENTION_HEADS,
                                     d_hid=FULLY_CONNECTED_LAYER_DIMENSION,
                                     nlayers=NUMBER_OF_LAYERS).to(device)

    opt = optim.Adam(params=transfencoder.parameters(), lr=LEARNING_RATE)
else:
    transfencoder = TransformerModel(ntoken=num_tokens,
                                     d_model=TOKEN_EMBEDDING_DIMENSION,
                                     nhead=NUMBER_OF_ATTENTION_HEADS,
                                     d_hid=FULLY_CONNECTED_LAYER_DIMENSION,
                                     nlayers=NUMBER_OF_LAYERS).to(device)
    transfencoder.load_state_dict(torch.load("./the_gpt_text_gen_model_dataset_pytorchimplem2_backup.pth"))
    opt = optim.Adam(params=transfencoder.parameters(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()

pytorch_total_params = sum(p.numel() for p in transfencoder.parameters())
pytorch_total_params_t = sum(p.numel() for p in transfencoder.parameters() if p.requires_grad)
print("model number of parameters:", pytorch_total_params)
print("model number of trainable parameters:", pytorch_total_params_t)


# src = torch.rand(BATCH_SIZE,SEQUENCE_LENGTH).type(torch.LongTensor)
# out = transfencoder(src)
# print(out.shape)


##################################################
# Auxilary function, generate string
##################################################
def generate_str(strr, lenn):
    strr_as_list = [charr for charr in strr]
    if len(strr) < SEQUENCE_LENGTH:
        strr_as_list = ["_" for _ in range(SEQUENCE_LENGTH - len(strr))] + strr_as_list
    elif len(strr) > SEQUENCE_LENGTH:
        strr_as_list = strr_as_list[len(strr) - SEQUENCE_LENGTH:]
    input_list = strr_as_list[:]

    result = strr[:]
    for i in range(lenn):
        input_tensor = []
        for m in range(len(input_list)):  # seq length 30
            input_tensor.append(token_to_index_dict[input_list[m]])
        input_tensor = autograd.Variable(torch.Tensor(input_tensor)).type(torch.LongTensor).unsqueeze(0).to(device)
        #         print("inputtensor shape",input_tensor.shape)
        outtt = transfencoder(input_tensor)
        _, pred = outtt.max(1)
        pred_token = "".join([tokens[inxx] for inxx in pred])
        input_list.pop(0)
        input_list.append(pred_token)
        result = result + pred_token
    return result


for epoch in range(1000):
    for batch_idx in range(100000,len(train_str) - SEQUENCE_LENGTH - 2,SEQUENCE_LENGTH):
        if batch_idx % 20 != 0:
            continue
        batch_x = []
        batch_y = []
        for seq_inside_batch in range(BATCH_SIZE):
            input_tensor = []
            for m in range(batch_idx + seq_inside_batch,
                           batch_idx + seq_inside_batch + SEQUENCE_LENGTH):  # seq length 30
                input_tensor.append(token_to_index_dict[train_str[m]])
            batch_x.append(input_tensor)
            batch_y.append(token_to_index_dict[train_str[batch_idx + seq_inside_batch + SEQUENCE_LENGTH]])

        bx = autograd.Variable(torch.Tensor(batch_x)).type(torch.LongTensor).to(device)
        by = autograd.Variable(torch.Tensor(batch_y)).type(torch.LongTensor).to(device)

        outtt = transfencoder(bx)
        #         print("outtt", outtt)
        #         print("outtt shape", outtt.shape)
        _, pred = outtt.max(1)
        #         print("pred", pred)

        losss = loss(outtt, by)
        transfencoder.zero_grad()
        losss.backward()
        opt.step()
        if batch_idx % 500 == 0:
            print(batch_idx)
            print("loss", losss)
            print("actual:", "".join([tokens[inxx] for inxx in by]))
            print("predicted:", "".join([tokens[inxx] for inxx in pred]))
            print("generated string:", generate_str("".join([tokens[inxx] for inxx in by]), 50))
            print()
            torch.save(transfencoder.state_dict(), "the_gpt_text_gen_model_dataset_pytorchimplem2.pth")
        if batch_idx % 5000 == 0:
            torch.save(transfencoder.state_dict(), "the_gpt_text_gen_model_dataset_pytorchimplem2_backup.pth")
        if batch_idx % 50000 == 0:
            torch.save(transfencoder.state_dict(), "the_gpt_text_gen_model_dataset_pytorchimplem2_backup_backup.pth")


##################################################
# Import the libraries
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
##################################################
FIRST_TIME_RUNNING = False
NEED_TOKEN_CALCULATION = False
TRAINING_DATA_PATH = "./xiaoshuo.txt"

TOKEN_EMBEDDING_DIMENSION = 210 # The dimention of vector representing a token
SEQUENCE_LENGTH = 95  # the length of input sequence
BATCH_SIZE = 150
FULLY_CONNECTED_LAYER_DIMENSION = 4*TOKEN_EMBEDDING_DIMENSION # dimention of the fully connected layer in the model
NUMBER_OF_ATTENTION_HEADS = 6  # the number of attention heads in the model
NUMBER_OF_LAYERS = 3  # the number of encoder block in the model
LEARNING_RATE = 1e-04
device = "cuda:0"

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

if NEED_TOKEN_CALCULATION:
    ### find all unique characters in the string
    for charr in train_str:
        if charr not in tokens:
            tokens.add(charr)
    tokens = list(tokens)

    # save the characters array
    with open('tokens_attention.pkl', 'wb') as file:
        pickle.dump(tokens, file)

    ### get the random token embedding and
    ### token to index in its array disctionary
    token_to_index_dict = {}
    for i in range(len(tokens)):
        token_to_index_dict[tokens[i]] = i
    token_embedding = np.random.rand(len(tokens), TOKEN_EMBEDDING_DIMENSION)

    ### Save the token embeddings
    with open('token_embedding_attention.pkl', 'wb') as file:
        pickle.dump(token_embedding, file)

    with open('token_to_index_dict_attention.pkl', 'wb') as file:
        pickle.dump(token_to_index_dict, file)

else:
    temp_file = open("tokens_attention.pkl", 'rb')
    tokens = pickle.load(temp_file)
    temp_file.close()

    temp_file = open("token_embedding_attention.pkl", 'rb')
    token_embedding = pickle.load(temp_file)
    temp_file.close()

    temp_file = open("token_to_index_dict_attention.pkl", 'rb')
    token_to_index_dict = pickle.load(temp_file)
    temp_file.close()

num_tokens = len(tokens)
print("Num of of tokens", str(len(tokens)))


##################################################
# Defining the model
##################################################
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        #         self.input_dim = input_dim
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    #         self.o = nn.Linear(input_dim*SEQUENCE_LENGTH, num_tokens)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        #         x =  self.o(x.view(-1,self.input_dim*SEQUENCE_LENGTH))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQUENCE_LENGTH):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [EncoderBlock(input_dim, num_heads, dim_feedforward, dropout=0.0) for _ in range(num_layers)])
        self.o = nn.Linear(input_dim * SEQUENCE_LENGTH, num_tokens)
        self.pos_enc = PositionalEncoding(d_model=input_dim)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.o(x.view(-1, self.input_dim * SEQUENCE_LENGTH))
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


mhead_att1 = ""
opt = ""
if FIRST_TIME_RUNNING:
    mhead_att1 = TransformerEncoder(num_layers=NUMBER_OF_LAYERS,
                                    input_dim=TOKEN_EMBEDDING_DIMENSION,
                                    dim_feedforward=FULLY_CONNECTED_LAYER_DIMENSION,
                                    num_heads=NUMBER_OF_ATTENTION_HEADS).to(device)

    opt = optim.Adam(params=mhead_att1.parameters(), lr=LEARNING_RATE)
else:
    mhead_att1 = TransformerEncoder(num_layers=NUMBER_OF_LAYERS,
                                    input_dim=TOKEN_EMBEDDING_DIMENSION,
                                    dim_feedforward=FULLY_CONNECTED_LAYER_DIMENSION,
                                    num_heads=NUMBER_OF_ATTENTION_HEADS).to(device)
    mhead_att1.load_state_dict(torch.load("./the_transformer_text_gen_modelv5_dataset2_v2_backup.pth"))
    opt = optim.Adam(params=mhead_att1.parameters(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()


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
            input_tensor.append(token_embedding[token_to_index_dict[input_list[m]]])
        input_tensor = autograd.Variable(torch.Tensor(input_tensor)).to(device)
        outtt = mhead_att1(input_tensor)
        _, pred = outtt.max(1)
        pred_token = "".join([tokens[inxx] for inxx in pred])
        input_list.pop(0)
        input_list.append(pred_token)
        result = result + pred_token
    return result


for epoch in range(1000):
    for batch_idx in range(150000,len(train_str) - SEQUENCE_LENGTH - 2,SEQUENCE_LENGTH):
        if batch_idx % 20 != 0:
            continue
        batch_x = []
        batch_y = []
        for seq_inside_batch in range(BATCH_SIZE):
            input_tensor = []
            for m in range(batch_idx + seq_inside_batch,
                           batch_idx + seq_inside_batch + SEQUENCE_LENGTH):  # seq length 30
                input_tensor.append(token_embedding[token_to_index_dict[train_str[m]]])
            batch_x.append(input_tensor)
            batch_y.append(token_to_index_dict[train_str[batch_idx + seq_inside_batch + SEQUENCE_LENGTH]])

        bx = autograd.Variable(torch.Tensor(batch_x)).type(torch.FloatTensor).to(device)
        by = autograd.Variable(torch.Tensor(batch_y)).type(torch.LongTensor).to(device)

        outtt = mhead_att1(bx)
        _, pred = outtt.max(1)

        losss = loss(outtt, by)
        mhead_att1.zero_grad()
        losss.backward()
        opt.step()
        if batch_idx % 500 == 0:
            print(batch_idx)
            print("loss", losss)
            print("actual:", "".join([tokens[inxx] for inxx in by]))
            print("predicted:", "".join([
                tokens[inxx] for inxx in pred]))
            print("generated string:", generate_str("".join([tokens[inxx] for inxx in by]), 50))
            print()
            torch.save(mhead_att1.state_dict(), "the_transformer_text_gen_modelv5_dataset2_v2.pth", _use_new_zipfile_serialization=False)
        if batch_idx % 50000 == 0:
            torch.save(mhead_att1.state_dict(), "the_transformer_text_gen_modelv5_dataset2_v2_backup.pth")
        if batch_idx % 500000 == 0:
            torch.save(mhead_att1.state_dict(), "the_transformer_text_gen_modelv5_dataset2_v2_backup_backup.pth")
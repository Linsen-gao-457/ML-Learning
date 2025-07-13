# here we want to implement 4 version of selfattention
# attention(Q,K,V) = softmax(Q * K/ sqrt(d_k))/ V

#-----------------
# version 1: simplified version
#-----------------
import math
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim:int = 128)-> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)
        
    def feedforward(self, X):
        # X's shape is (batch_size, seq_len, embedding_dim)
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        # QKV's shape is (batch_size, seq_len, embedding_dim)
        # atention_value (batch_size, seq, seq)
        atn_val = Q@K.transpose(-1,-2)
        attention_weight = torch.softmax(
            atn_val/math.sqrt(self.hidden_dim), dim = -1
        )
        # output (bathch_size, seq, seq)
        output = attention_weight @ V
        print(output)
        return output
        
X = torch.rand(3,2,4)
print(X)

self_attn_net = SelfAttentionV1(hidden_dim=4)
output = self_attn_net.feedforward(X)

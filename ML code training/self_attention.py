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
        # X's shape is (batch_size, seq_len, hidden_dim)
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        # QKV's shape is (batch_size, seq_len, hidden_dim)
        # atention_value (batch_size, seq, seq)
        atn_val = Q@K.transpose(-1,-2)
        attention_weight = torch.softmax(
            atn_val/math.sqrt(self.hidden_dim), dim = -1
        )
        print(attention_weight)
        # output (bathch_size, seq, seq)
        output = attention_weight @ V
        print(output)
        return output
        
# X = torch.rand(3,2,4)
# print(X)

# self_attn_net = SelfAttentionV1(hidden_dim=4)
# output = self_attn_net.feedforward(X)


#---------------
# version 2: combine QKV proj
#---------------

class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim:int = 128)-> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prj = nn.Linear(hidden_dim, hidden_dim*3)
        
    def feedforward(self, X):
        # X's shape (batch_size, seq_len, hidden_dim)
        QKV = self.prj(X)
        # QKV's shape(batch_size, seq_len, hidden_dim *3)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attn_val = Q@K.transpose(1,2)
        attn_weight = torch.softmax(attn_val/math.sqrt(self.hidden_dim), dim=-1)
        print(attn_weight)
        output = attn_weight @ V
        return output
    
    
X = torch.rand(2,3,4)
net = SelfAttentionV2(4)
net.feedforward(X)
#------------
# version 3: 
#-----------------

class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim,3*hidden_dim)

    def FeedForward(self, X):
        # X's shape (batch_size, seq_len, hidden_dim)
        QKV = self.proj(X)
        # QKV's shape is (batch_size, seq_len, 3*hidden_dim)
        Q, K, V = torch.split(QKV, dim=-1, split_size_or_sections=self.hidden_dim)
        atten_val = Q @ K.transpose(-1,-2)
        atten_weight = torch.softmax(atten_val/math.sqrt(self.hidden_dim), dim= -1)
        output = atten_weight @ V
        return output

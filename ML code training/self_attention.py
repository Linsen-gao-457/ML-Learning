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
    
    
# X = torch.rand(2,3,4)
# net = SelfAttentionV2(4)
# net.feedforward(X)

#------------
# version 3: add more detail: 1. dropout 2. attention_mask 3. output matrice (optional)
#------------
class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim,3*hidden_dim)
        self.attention_dropout = nn.Dropout(p=dropout_rate)
    def FeedForward(self, X, attention_mask = None):
        # X's shape (batch_size, seq_len, hidden_dim)
        QKV = self.proj(X)
        # QKV's shape is (batch_size, seq_len, 3*hidden_dim)
        Q, K, V = torch.split(QKV, dim=-1, split_size_or_sections=self.hidden_dim)
        atten_val = Q @ K.transpose(-1,-2)
        # atten_val's shape (batch_size, seq, seq)
        if attention_mask is not None: 
            atten_val = atten_val.masked_fill(
                attention_mask == 0, float("-1e20")
            )
        print(atten_val)
        atten_weight = torch.softmax(atten_val/math.sqrt(self.hidden_dim), dim= -1)   
        print(atten_weight)
        atten_weight = self.attention_dropout(atten_weight)
        output = atten_weight @ V
        return output

# X = torch.rand(3,4,2)
# print(X.shape)
# mask = torch.tensor(
#     [
#         [1,1,1,0],
#         [1,1,0,0],
#         [1,0,0,0]
#     ]
# )

# mask = mask.unsqueeze(dim=1).repeat(1,4,1)
# print(mask.shape)
# print(mask)

# net = SelfAttentionV3(2)
# net.FeedForward(X, attention_mask=mask)


#------------
# version4: version for interview
#------------

class SelfAttentionInterview(nn.Module):
    def __init__(self, hidden_dim:int, dropout_rate: float = 0.1, *args, **kwargs)->None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.Q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.K_proj = nn.Linear(hidden_dim,hidden_dim)
        self.V_proj = nn.Linear(hidden_dim,hidden_dim)
        self.attndropout = nn.Dropout(p=dropout_rate)

    def feedforward(self, X, mask=None):
        Q = self.Q_proj(X)
        K = self.K_proj(X)
        V = self.V_proj(X)
        attn_val = Q @ K.transpose(-1,-2)
        if mask is not None:
            attn_val = attn_val.masked_fill(
                mask == 0, float("-inf")
            )
        # attn_val's size (batch_size, hidden_dim, hidden_dim)
        attn_weight = self.attndropout(attn_weight)
        attn_weight = torch.softmax(attn_val/math.sqrt(self.hidden_dim), dim=-1)
        output = attn_weight @ V
        print(output)
        return output
    
    
# X = torch.rand(2,4,2)
# net = SelfAttentionInterview(hidden_dim=2)

# mask = torch.tensor(
#     [
#         [1,1,1,0],
#         [1,1,0,0]
#     ]
# )
# mask = mask.unsqueeze(1).repeat(1,4,1)
# net.feedforward(X=X,mask=mask)


# class MultiHeadAttentionFormal(nn.Module):
#     def __init__(self, hidden_dim:int = 128, head_num:int= 8, attention_dropout_rate = 0.1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.hidden_dim = hidden_dim
#         self.head_num = head_num
#         self.head_dim = hidden_dim//head_num
#         self.Q_proj = nn.Linear(hidden_dim,hidden_dim)
#         self.K_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.V_proj = nn.Linear(hidden_dim,hidden_dim)
#         self.output_proj = nn.Linear(hidden_dim,hidden_dim)
#         self.attention_dropout = nn.Dropout(p = attention_dropout_rate)

#     def Forward(self, X, mask= None):
#         # X's shape (batch_size, seq_len, hidden_dim)
#         batch_size, seq_size, _ = X.size()
#         Q = self.Q_proj(X)
#         K = self.K_proj(X)
#         V = self.V_proj(X) 
#         # (b,s,hidden_dim)-> (b, head_num, s, head_dim)
#         q_state = Q.view(batch_size, seq_size, self.head_num, self.head_dim).transpose(1,2)
#         k_state = K.view(batch_size, seq_size, self.head_num, self.head_dim).transpose(1,2)
#         v_state = V.view(batch_size, seq_size, self.head_num, self.head_dim).transpose(1,2)
#         # atten_val's shape is (batch_size, head_num, seq_len, seq_len)
#         atten_val = q_state @ k_state.transpose(-1,-2)/math.sqrt(self.head_dim)
        
#         if mask is not None:
#             atten_val = atten_val.masked_fill(
#                 atten_val == 0, float("-inf")
#             )
#         atten_val = self.attention_dropout(atten_val)
#         atten_weight = torch.softmax(atten_val, dim=-1)
#         output_mid = atten_weight @ v_state # (b, head_num, seq, head_dim) -> (b,s, hidden_dim)
#         output_mid = output_mid.transpose(1,2).contiguous()
#         output_mid = output_mid.view(batch_size, seq_size, -1)
#         output = self.output_proj(output_mid)
#         print(output)
#         return output

# X = torch.rand(3,2,128)
# net = MultiHeadAttentionFormal(hidden_dim=128, head_num=8)

# mask = (
# torch.tensor(
#     [
#         [0,1],
#         [0,0],
#         [1,0]
#     ]
# ).unsqueeze(1).unsqueeze(2).expand(3,8,2,2)
# )
# print(mask.size())
# net.Forward(X=X,mask=mask)


class MultiHeadatten(nn.Module):
    def __init__(self, hidden_dim:int =128, head_num:int =8, attention_droprate = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim//head_num
        self.Q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.K_proj = nn.Linear(hidden_dim,hidden_dim)
        self.V_proj = nn.Linear(hidden_dim,hidden_dim)
        self.Out_proj = nn.Linear(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(p=attention_droprate)

    def Forward(self, X, mask = None):
        # X's shape(batch_size, seq_len, hidden_dim)
        # prj (batch_size, seq_len, hidden_dim) -> multi_head (batch_size, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = X.size()
        Q = self.Q_proj(X)
        K = self.K_proj(X)
        V = self.V_proj(X)
        Q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        K_state = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        V_state = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        # atten_val's shape (batch_size, head_num, seq_len, seq_len)
        atten_val = Q_state @ K_state.transpose(-1,-2)/math.sqrt(self.head_dim)
        if mask is not None:
            atten_val = atten_val.masked_fill(
                atten_val == 0, float("-inf")
            )
        self.dropout(atten_val)
        atten_weight = torch.softmax(atten_val, dim=-1)
        print(atten_weight.shape)
        # outpus's shape is (batch_size, head_num, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        pre_output = atten_weight @ V_state
        pre_output = pre_output.transpose(1,2).contiguous()
        pre_output=pre_output.view(batch_size, seq_len, -1)
        output = self.Out_proj(pre_output)
        return output
    
X = torch.rand(3,2,128)
net = MultiHeadatten()
# mask size(batch_size=3, head_num= 8, seq_len= 2, seq_len=2)
mask = (torch.Tensor(
    [
        [1,1],
        [1,0],
        [1,0]
    ]
)).unsqueeze(1).unsqueeze(2).expand(3,8,2,2)
net.Forward(X=X, mask=mask)
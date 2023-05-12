import torch
import torch.nn as nn
import torch.nn.functional as F

class QKVAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.2):
        super().__init__()
        self.qkv = nn.Linear(in_channels, out_channels * 3)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x.view(B, C, -1).permute(2, 0, 1))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # self.attention() expects (seq_len, batch, features), or (W*H, B, C)
        #print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        out, _ = self.attention(q, k, v)
        # out: (W*H, B, C) or (seq_len, batch, features)
        return out.permute(1, 2, 0).view(B, C, H, W)

class CrossAttention(nn.Module):
    def __init__(self, in_channels_q, in_channels_kv, out_channels, num_heads, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(in_channels_q, out_channels)
        self.key_value = nn.Linear(in_channels_kv, out_channels * 2)
        self.attention = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout,
            kdim=in_channels_kv, vdim=in_channels_kv
            )
        
    def forward(self, x_query, x_key_value):
        B_q, C_q, H_q, W_q = x_query.size()
        B_kv, C_kv, H_kv, W_kv = x_key_value.size()
        #print(f"query: {x_query.shape}, key_value: {x_key_value.shape}")
        #print(f"query_view: {x_query.view(B_q, C_q, -1).shape}")
        query = self.query(x_query.view(B_q, C_q, -1).permute(2, 0, 1)) # (W_q*H_q, B_q, C_q)
        key_value = self.key_value(x_key_value.view(B_kv, C_kv, -1).permute(2, 0, 1))
        key, value = torch.chunk(key_value, 2, dim=-1)
        
        out, _ = self.attention(query, key, value)
        #print(out.shape)
        return out.permute(1, 2, 0).view(B_q, -1, H_q, W_q)


class InvertedCrossAttention(nn.Module):
    def __init__(self,
                 in_channels_q, # query feature map W_q*H_q (used as seq_len)
                 in_channels_kv, # key/value feature map W_kv*H_kv (used as seq_len)
                 num_heads):
        super().__init__()
        self.query = nn.Linear(in_channels_q, in_channels_kv)
        self.key_value = nn.Linear(in_channels_kv, in_channels_kv * 2)
        self.attention = nn.MultiheadAttention(in_channels_kv, num_heads)

    def forward(self, x_query, x_key_value):
        B_q, C_q, H_q, W_q = x_query.size()
        B_kv, C_kv, H_kv, W_kv = x_key_value.size()
        query = self.query(x_query.view(B_q, C_q, -1).permute(1, 0, 2)) # (C_q, B_q, W_q*H_q)
        key_value = self.key_value(x_key_value.view(B_kv, C_kv, -1).permute(1, 0, 2)) # (C_kv, B_kv, W_kv*H_kv*2)
        key, value = torch.chunk(key_value, 2, dim=-1)
        #print(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")
        # out: (C_kv, B_q, W_q*H_q) or (seq_len, batch, features)
        out, _ = self.attention(query, key, value)
        return out.permute(1, 0, 2).view(B_kv, C_kv, H_kv, W_kv)
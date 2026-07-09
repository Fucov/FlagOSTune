import math
import torch
import torch.nn as nn

def sig_attn(q,k,v,mask):
    d=q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d)
    if mask is not None:
        scores=scores.masked_fill(mask==0,float("-inf"))
    attn=torch.softmax(scores,dim=-1)
    out=torch.matmul(attn,v)
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super().__init_()
        assert hidden_size % num_heads ==0

        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads

        self.q_proj=nn.Linear(hidden_size,hidden_size)
        self.k_proj=nn.Linear(hidden_size,hidden_size)
        self.v_proj=nn.Linear(hidden_size,hidden_size)
        self.out_proj=nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,mask=None):
        B,T,C = x.shape
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        # [B,T,C] -> [B,nums_heads,T,head_dim]
        # B: batch size, T: sequence length, C: hidden size
        # 为什么transpose(1,2)？因为我们希望将num_heads维度放在第二个位置，方便后续的矩阵乘法操作。最终的形状是[B, num_heads, T, head_dim]，这样每个头可以独立计算注意力分数。
        q=q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        scores=torch.matmul(q,k.transpose(-2,-1)) /math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 3:
                scores =scores.masked_fill(mask==0, float("-inf"))
        attn = torch.softmax(scores ,dim =-1)
        out = torch.matmul(attn,v)

        # 为什么contiguous()？因为在调用view()之前，我们需要确保张量在内存中是连续的。transpose()操作可能会导致张量在内存中不连续，因此我们使用contiguous()来创建一个新的连续张量，以便后续的view()操作能够正确地重新排列数据。
        out = out.transpose(1,2).contiguous().view(B,T,C)
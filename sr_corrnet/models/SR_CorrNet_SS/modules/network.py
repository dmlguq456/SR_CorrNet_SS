import torch as th
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding



class SwiGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.silu = nn.SiLU()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * self.silu(x2)



class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.kv = nn.Linear(emb_dim, attention_dim * 2, bias=False)
        self.q = nn.Linear(emb_dim, attention_dim, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        
        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, query, key_value):
        # get query, key, and value
        query = self.get_q(query)
        key, value = self.get_kv(key_value)

        # rotary positional encoding
        if self.rope != None:
            query, key = self.apply_rope(query, key)

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with th.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_kv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.kv(input).reshape(n_batch, seq_len, 2, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        key, value = x[..., 0, :], x[..., 1, :]
        return key, value

    def get_q(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.q(input).reshape(n_batch, seq_len, 1, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query = x[..., 0, :]
        return query


    @th.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        
        if flash_attention:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, attn_mask=None):
        # get query, key, and value
        # attn mask: B, H, L, L or B, 1, L, L)
        query, key, value = self.get_qkv(input)
        

        # rotary positional encoding
        if self.rope != None:
            query, key = self.apply_rope(query, key)
        
        
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with th.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @th.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

    
    
class MHSA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float, rope=None, flash_attention=False):
        super().__init__()
        self.layernorm = nn.RMSNorm(d_model)
        self.block = MultiHeadSelfAttention(d_model, d_model, n_head, dropout_rate, rope, flash_attention)

    def forward(self, x: th.Tensor, attn_mask=None):
        """
        Compute encoded features.
            :param th.Tensor x: encoded source features (batch, max_time_in, size)
            :param th.Tensor attn_mask: mask for attention (batch, max_time_in)
            :rtype: Tuple[th.Tensor, th.Tensor]
        """
        y = self.layernorm(x)
        y = self.block(y, attn_mask=attn_mask)

        return x + y
    
    
class MHCA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float, rope=None, freq_pe=None):
        super().__init__()
        self.layernorm = nn.RMSNorm(d_model)
        self.block = MultiHeadCrossAttention(d_model, d_model, n_head, dropout_rate, rope, freq_pe)
    
    def forward(self, x: th.Tensor, kv: th.Tensor):
        """
        Compute encoded features.
            :param th.Tensor x: encoded source features (batch, max_time_in, size)
            :param th.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[th.Tensor, th.Tensor]
        """
        y = self.layernorm(x)
        y = self.block(y, kv)
        
        return x + y


class ConvFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, kernel_size: int, dropout_rate: float,):
        super().__init__()
        self.layernorm = nn.RMSNorm(d_model)
        if dropout_rate > 0:
            self.net = nn.Sequential(
                nn.Conv1d(d_model, d_hidden*2, kernel_size, padding='same'),
                SwiGLU(dim=-2),
                nn.Dropout(dropout_rate),
                nn.Conv1d(d_hidden, d_model, kernel_size, padding='same'),
                nn.Dropout(dropout_rate)
                )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(d_model, d_hidden*2, kernel_size, padding='same'),
                SwiGLU(dim=-2),
                nn.Conv1d(d_hidden, d_model, kernel_size, padding='same'),
            )

    def forward(self, x):
        y = self.layernorm(x)
        y = y.permute(0, 2, 1)
        y = self.net(y)
        y = y.permute(0, 2, 1)
        return x + 0.5*y
    


class FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.RMSNorm(d_model)
        if dropout_rate > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, 2*d_hidden),
                SwiGLU(dim=-1),
                nn.Dropout(dropout_rate),
                nn.Linear(d_hidden, d_model),
                nn.Dropout(dropout_rate)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, 2*d_hidden),
                SwiGLU(dim=-1),
                nn.Linear(d_hidden, d_model),
            )

    def forward(self, x):
        y = self.layernorm(x)
        y = self.net(y)
        return x + y


    
class TransBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_head: int, kernel_size: int, dropout_rate: float, rope=None):
        super().__init__()
        self.ffn_1 = ConvFFN(d_model, d_hidden, kernel_size, dropout_rate) 
        self.sa = MHSA(d_model, n_head, dropout_rate, rope=rope)
        self.ffn_2 = ConvFFN(d_model, d_hidden, kernel_size, dropout_rate) 

    def forward(self, x: th.tensor):
        #* B, L, C
        x = self.ffn_1(x)
        x = self.sa(x)
        x = self.ffn_2(x)
        return x

class TransDecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_head: int, dropout_rate: float):
        super().__init__()
        self.ca = MHCA(d_model, n_head, dropout_rate)
        self.sa = MHSA(d_model, n_head, dropout_rate)
        self.ffn = FFN(d_model, d_hidden, dropout_rate)

        # Cached causal mask to avoid re-creating every forward pass
        self._cached_mask = None
        self._cached_mask_size = 0

    def forward(self, x: th.tensor, kv: th.tensor):
        #* B, L, C
        batch_size, seq_len = x.shape[:2]

        # Reuse cached causal mask when possible, only recreate when seq_len grows
        if seq_len > self._cached_mask_size:
            self._cached_mask = th.tril(th.ones(seq_len, seq_len, dtype=th.bool))
            self._cached_mask_size = seq_len
        causal_mask = self._cached_mask[:seq_len, :seq_len].to(x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)  # (batch, 1, seq_len, seq_len)
        
        x = self.ca(x, kv)
        x = self.sa(x, attn_mask=causal_mask)
        x = self.ffn(x)
        return x



class CS_TransBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_head: int, dropout_rate: float):
        super().__init__()
        self.block = nn.ModuleDict({
            'sa': MHSA(d_model=d_model, n_head=n_head, dropout_rate=dropout_rate, flash_attention=True),
            'ffn': FFN(d_model=d_model, d_hidden=d_hidden, dropout_rate=dropout_rate)
        })

    def forward(self, x: th.tensor, attn_mask=None):
        #* B, L, C
        x = self.block['sa'](x, attn_mask=attn_mask)
        x = self.block['ffn'](x)
        
        return x

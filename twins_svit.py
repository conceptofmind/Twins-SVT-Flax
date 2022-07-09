import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

import numpy as np

from typing import Callable

from einops import rearrange, repeat

from einops import rearrange

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

class Residual(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        eps = self.eps
        g = self.param('g', nn.initializers.ones, [1, 1, 1, self.dim])
        b = self.param('b', nn.initializers.zeros, [1, 1, 1, self.dim])

        var = jnp.var(x, axis = -1, keepdims = True)
        mean = jnp.mean(x, axis = -1, keepdims = True)

        x = (x - mean) / jnp.sqrt((var + eps)) * g + b

        return x

class MLP(nn.Module):
    dim: int 
    mult: int = 4 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features = self.dim * self.mult, kernel_size = (1, 1), strides = (1, 1))(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x)
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1))(x)
        x = nn.Dropout(rate = self.dropout)(x)
        return x

class PatchEmbedding(nn.Module):
    dim_out: int 
    patch_size: int

    @nn.compact
    def __call__(self, fmap):

        dim_out = self.dim_out
        patch_size = self.patch_size
        proj = nn.Conv(features = dim_out, kernel_size = (1, 1), strides = (1, 1))

        p = patch_size
        fmap = rearrange(fmap, 'b (h p1) (w p2) c -> b h w (c p1 p2)', p1 = p, p2 = p)
        x = proj(fmap)

        return x

class PEG(nn.Module):
    dim: int
    kernel_size: int = 3
        
    @nn.compact
    def __call__(self, x):
        proj = Residual(nn.Conv(features = self.dim, 
                                kernel_size = (self.kernel_size, self.kernal_size), 
                                strides = (1, 1), 
                                padding = 'SAME', 
                                feature_group_count = self.dim))
        x = proj(x)
        return x

class LocalAttention(nn.Module): 
    dim: int 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.0 
    patch_size: int = 7

    @nn.compact
    def __call__(self, fmap):

        inner_dim = self.dim_head * self.heads
        patch_size = self.patch_size
        heads = self.heads
        scale = self.dim_head ** -0.5

        to_q = nn.Conv(features = inner_dim, kernel_size = (1, 1), strides = (1, 1), use_bias=False)
        to_kv = nn.Conv(features = inner_dim * 2, kernel_size = (1, 1), strides = (1, 1), use_bias=False)

        to_out = nn.Sequential([
            nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1)),
            nn.Dropout(rate = self.dropout)
        ])

        b, x, y, n = fmap.shape
        h = heads
        p = patch_size
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b (x p1) (y p2) c -> (b x y) p1 p2 c', p1 = p, p2 = p)
        q = to_q(fmap)
        kv = to_kv(fmap)
        k, v = jnp.split(kv, 2, axis = -1)

        q, k, v = map(lambda t: rearrange(t, 'b p1 p2 (h d) -> (b h) (p1 p2) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (x p1) (y p2) (h d) ', h = h, x = x, y = y, p1 = p, p2 = p)
        out = to_out(out)

        return out

class GlobalAttention(nn.Module): 
    dim: int 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.0 
    k: int = 7

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head * self.heads
        heads = self.heads
        scale = self.dim_head ** -0.5

        to_q = nn.Conv(features = inner_dim, kernel_size = (1, 1), use_bias = False)
        to_kv = nn.Conv(features = inner_dim * 2, kernel_size = (k, k), strides = (k, k), use_bias = False)

        to_out = nn.Sequential([
            nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1)),
            nn.Dropout(rate = self.dropout)
        ])

        b, _, y, n = x.shape
        h = heads

        q = to_q(x)
        kv = to_kv(x)
        k, v = jnp.split(kv, 2, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b x y (h d)', h = h, y = y)
        out = to_out(out)
        return out

class Transformer(nn.Module):
    dim: int 
    depth: int 
    heads: int = 8 
    dim_head: int = 64 
    mlp_mult: int = 4 
    local_patch_size: int = 7 
    global_k: int = 7 
    dropout: float = 0.0 
    has_local: bool = True

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                Residual(PreNorm(self.dim, LocalAttention(self.dim, self.heads, self.dim_head, dropout = self.dropout, patch_size = self.local_patch_size))) if self.has_local else IdentityLayer(),
                Residual(PreNorm(self.dim, MLP(self.dim, self.mlp_mult, dropout = self.dropout))) if self.has_local else IdentityLayer(),
                Residual(PreNorm(self.dim, GlobalAttention(self.dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout, k = self.global_k))),
                Residual(PreNorm(self.dim, MLP(self.dim, self.mlp_mult, dropout = self.dropout)))
            ])

        for local_attn, ff1, global_attn, ff2 in layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)

        return x

class TwinsSVT(nn.Module):
    num_classes: int
    s1_emb_dim: int = 64
    s1_patch_size: int = 4
    s1_local_patch_size: int = 7
    s1_global_k: int = 7
    s1_depth: int = 1
    s2_emb_dim: int = 128
    s2_patch_size: int = 2
    s2_local_patch_size: int = 7
    s2_global_k: int = 7
    s2_depth: int = 1
    s3_emb_dim: int = 256
    s3_patch_size: int = 2
    s3_local_patch_size: int = 7
    s3_global_k: int = 7
    s3_depth: int = 5
    s4_emb_dim: int = 512
    s4_patch_size: int = 2
    s4_local_patch_size: int = 7
    s4_global_k: int = 7
    s4_depth: int = 4
    peg_kernel_size: int = 3
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, **kwargs):

        kwargs = dict(locals())

        layers = nn.Sequential([])

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'
            print(config)
            dim_next = config['emb_dim']

            layers.append(nn.Sequential([
                PatchEmbedding(dim_out=dim_next, patch_size=config['patch_size']),
                Transformer(dim=dim_next, depth=1, local_patch_size=config['local_patch_size'],
                            global_k=config['global_k'], dropout = self.dropout, has_local=not is_last),
                PEG(dim=dim_next, kernel_size = self.peg_kernel_size),
                Transformer(dim=dim_next, depth=config['depth'], local_patch_size=config['local_patch_size'],
                            global_k=config['global_k'], dropout = self.dropout, has_local=not is_last)
            ]))

        layers += nn.Sequential([
            nn.avg_pool(),
            nn.Dense(features = self.num_classes)
        ])

        x = svt_layers(x)
        return x

if __name__ == "__main__":

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = TwinsSVT(
        num_classes = 1000,       # number of output classes
        s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
        s1_patch_size = 4,        # stage 1 - patch size for patch embedding
        s1_local_patch_size = 7,  # stage 1 - patch size for local attention
        s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        s2_emb_dim = 128,         # stage 2 (same as above)
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,         # stage 3 (same as above)
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,         # stage 4 (same as above)
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,      # positional encoding generator kernel size
        dropout = 0.              # dropout
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")
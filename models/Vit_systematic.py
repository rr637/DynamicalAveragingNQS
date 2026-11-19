
# Vision Transformer architecture adapted from ptvmc-systematic-study
# (L. Gravina, F. Vicentini), https://github.com/NeuralQXLab/ptvmc-systematic-study,
# packages/ptvmc/_src/nets/ViT/. See: Gravina, Savona, Vicentini,
# "Neural Projected Quantum Dynamics: a systematic study", arXiv:2410.10720.

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax._src import dtypes
from einops import rearrange


def extract_patches2d_rect(x, b, Lx, Ly):
    """
    Extract b x b patches from rectangular Lx x Ly input.

    Input:
        x: shape (batch, Lx*Ly)
    Output:
        x: shape (batch, n_patches, b*b)
    """
    batch = x.shape[0]

    assert Lx % b == 0, f"Lx={Lx} must be divisible by b={b}"
    assert Ly % b == 0, f"Ly={Ly} must be divisible by b={b}"
    assert x.shape[1] == Lx * Ly, f"x.shape[1]={x.shape[1]} but Lx*Ly={Lx*Ly}"

    Lx_eff = Lx // b
    Ly_eff = Ly // b

    x = x.reshape(batch, Lx_eff, b, Ly_eff, b)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, Lx_eff * Ly_eff, b * b)

    return x


def custom_uniform(scale=1e-2, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0 * random.uniform(key, shape, dtype) - 1.0) * scale

    return init


@partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0, None, None), out_axes=1)
def roll2d_rect(mat, i, j, Lx_eff, Ly_eff):
    """
    Rectangular translation tying.

    mat: shape (heads, Lx_eff * Ly_eff)

    Returns translated copies with shape:
        (heads, Lx_eff, Ly_eff, Lx_eff * Ly_eff)
    before final reshape.
    """
    mat = mat.reshape(mat.shape[0], Lx_eff, Ly_eff)
    mat = jnp.roll(jnp.roll(mat, i, axis=-2), j, axis=-1)
    return mat.reshape(mat.shape[0], -1)


@partial(jax.vmap, in_axes=(None, 0, None))
def _compute_attn(J, x, h):
    x = rearrange(x, "L_eff (h d_eff) -> L_eff h d_eff", h=h)
    x = rearrange(x, "L_eff h d_eff -> h L_eff d_eff")
    x = jnp.matmul(J, x)
    x = rearrange(x, "h L_eff d_eff -> L_eff h d_eff")
    x = rearrange(x, "L_eff h d_eff -> L_eff (h d_eff)")
    return x


class FMHA(nn.Module):
    d_model: int
    h: int
    L_eff: int
    Lx_eff: int
    Ly_eff: int
    transl_invariant: bool = True

    def setup(self):
        assert self.d_model % self.h == 0, "d_model must be divisible by heads"
        assert self.Lx_eff * self.Ly_eff == self.L_eff

        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

        if self.transl_invariant:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.L_eff) ** 0.5),
                (self.h, self.L_eff),
                jnp.float64,
            )

            self.J = roll2d_rect(
                self.J,
                jnp.arange(self.Lx_eff),
                jnp.arange(self.Ly_eff),
                self.Lx_eff,
                self.Ly_eff,
            )

            self.J = self.J.reshape(self.h, self.L_eff, self.L_eff)

        else:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.L_eff) ** 0.5),
                (self.h, self.L_eff, self.L_eff),
                jnp.float64,
            )

        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.v(x)
        x = _compute_attn(self.J, x, self.h)
        x = self.W(x)
        return x


class Embed(nn.Module):
    d_model: int
    b: int
    extract_patches: Callable

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)
        return x


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    Lx_eff: int
    Ly_eff: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            h=self.h,
            L_eff=self.L_eff,
            Lx_eff=self.Lx_eff,
            Ly_eff=self.Ly_eff,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    self.expansion_factor * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    Lx_eff: int
    Ly_eff: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                L_eff=self.L_eff,
                Lx_eff=self.Lx_eff,
                Ly_eff=self.Ly_eff,
                expansion_factor=self.expansion_factor,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class OutputHead(nn.Module):
    d_model: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm2 = nn.LayerNorm(
            use_scale=True,
            use_bias=True,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True,
            use_bias=True,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        x = self.out_layer_norm(x.sum(axis=-2))

        amp = self.norm2(self.output_layer0(x))
        sign = self.norm3(self.output_layer1(x))

        z = amp + 1j * sign
        return jnp.sum(log_cosh(z), axis=-1)


class ViT(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    b: int
    Lx: int
    Ly: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        assert self.Lx % self.b == 0, f"Lx={self.Lx} must be divisible by b={self.b}"
        assert self.Ly % self.b == 0, f"Ly={self.Ly} must be divisible by b={self.b}"

        self.Lx_eff = self.Lx // self.b
        self.Ly_eff = self.Ly // self.b
        self.L_eff = self.Lx_eff * self.Ly_eff

        self.patches_and_embed = Embed(
            self.d_model,
            self.b,
            extract_patches=partial(
                extract_patches2d_rect,
                Lx=self.Lx,
                Ly=self.Ly,
            ),
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            Lx_eff=self.Lx_eff,
            Ly_eff=self.Ly_eff,
            expansion_factor=self.expansion_factor,
            transl_invariant=self.transl_invariant,
        )

        self.output = OutputHead(self.d_model)

    def __call__(self, x):
        x = self.patches_and_embed(x)
        x = self.encoder(x)
        z = self.output(x)
        return z
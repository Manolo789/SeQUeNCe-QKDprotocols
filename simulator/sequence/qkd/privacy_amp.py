# -*- coding: utf-8 -*-
"""
==================================================================
Privacy amplification (post-processing layer 3) -- License
==================================================================

Copyright © 2026 Manolo789 -- https://github.com/Manolo789/SeQUeNCe-QKDprotocols

All rights reserved. (Same MIT-style terms as the rest of the extension;
see error_correction.py for the full text.)

==================================================================

Layer 3 of the classical post-processing stack: PRIVACY AMPLIFICATION.

Removes Eve's residual information from the reconciled key by compressing
it with a randomly chosen function of a two-universal family, i.e. a
quantum-proof strong randomness extractor (Wolf, "Quantum Key
Distribution", Sect. 4.2.3, Definition 4.8 and the Quantum Leftover Hash
Lemma, Lemma 4.9): the output of length ``l`` (bounded by the
entropy-estimation layer, Eq. 4.44) is ``eps_pa``-close to a uniform key
independent of the seed and of Eve's system.

Choreography (1 classical message per key):

    Alice --PA_SEED(seed, l)--> Bob

Alice draws the extractor seed from her node RNG, applies the extractor to
her reconciled string and announces (seed, l) publicly -- a STRONG
extractor keeps the output independent of the seed, so the announcement is
safe. Bob applies the same function to his (identical, post error
correction) string; both obtain the same final key of ``l`` bits.

Implementation note: the extractor IS a Toeplitz hash over GF(2) -- the
canonical two-universal family (Wolf, Example 4.3). The public seed
expands, through a seeded PRNG, into the ``n + l - 1`` diagonal bits
``d`` that define the l x n binary Toeplitz matrix ``T[i][j] =
d[i - j + n - 1]``, and the final key is ``y = T x mod 2``. The
matrix-vector product is evaluated as a convolution (``y_i =
(x * d)[n - 1 + i] mod 2``), computed directly for short keys and via
FFT for long ones, so the cost is O(n log n) instead of O(n l).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from ..message import Message
from ..protocol import StackProtocol
from .error_correction import bits_to_int

if TYPE_CHECKING:
    from ..topology.node import Node


def two_universal_extract(bits: list, seed: int, out_len: int) -> list:
    """Compress ``bits`` to ``out_len`` bits with a GF(2) Toeplitz hash.

    The seed deterministically generates the ``n + out_len - 1`` diagonal
    bits of the Toeplitz matrix (both parties expand the same public seed
    into the same matrix), and the output is the GF(2) matrix-vector
    product, evaluated as the middle slice of the binary convolution of
    the input with the diagonal:

        y_i = sum_j T[i][j] x_j = (x * d)[n - 1 + i]   (mod 2).

    For inputs above ~4096 bits the convolution runs through the FFT; the
    linear sums are bounded by ``n`` (< 2^53), so rounding the real FFT
    back to integers is exact for every campaign key size.

    Args:
        bits (list[int]): reconciled key (x, length n).
        seed (int): public extractor seed (chosen by Alice).
        out_len (int): output length l from the entropy-estimation layer.

    Returns:
        list[int]: final key bits (empty when ``out_len`` <= 0).
    """
    n = len(bits)
    if out_len <= 0 or n == 0:
        return []
    out_len = min(out_len, n)
    rng = np.random.default_rng(int(seed) & ((1 << 128) - 1))
    diag = rng.integers(0, 2, size=n + out_len - 1, dtype=np.int64)
    x = np.asarray(bits, dtype=np.int64)

    if n <= 4096:
        conv = np.convolve(x, diag)
    else:
        m = n + diag.size - 1               # full convolution length
        nfft = 1 << (m - 1).bit_length()
        conv = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(diag, nfft),
                            nfft)[:m]
        conv = np.rint(conv).astype(np.int64)

    y = (conv[n - 1:n - 1 + out_len] & 1).astype(int)
    return y.tolist()


class PAMsgType(Enum):
    """Message types of the privacy-amplification layer."""
    PA_SEED = auto()     # Alice -> Bob: extractor seed + output length


class PAMessage(Message):
    """Classical message exchanged by the privacy-amplification layer."""

    def __init__(self, msg_type: PAMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = PrivacyAmplification
        self.kwargs = kwargs


class PrivacyAmplification(StackProtocol):
    """Two-universal-hash compression of the reconciled key (layer 3).

    Attributes:
        role (int): 0 = Alice, 1 = Bob.
        final_len_list (list[int]): per-key output length l.
        final_keys (list[int | None]): per-key final key (packed int),
            None for aborted keys (l = 0).
    """

    def __init__(self, owner: "Node", name: str):
        super().__init__(owner, name)
        self.role = -1
        self.another: "PrivacyAmplification" = None
        self.peer_node: str = None
        self.peer_name: str = None

        self.final_len_list: list = []
        self.final_keys: list = []

        self._bob_buffer: dict = {}     # key_index -> reconciled bits
        self._bob_pending: dict = {}    # key_index -> PA_SEED message

    # ------------------------------------------------------------------
    def push(self, **kwargs):
        self._push(**kwargs)

    def pop(self, bits: list = None, key_index: int = 0,
            secret_len: int = None, **kwargs):
        """Receive one reconciled key from the entropy-estimation layer."""
        if self.role == 0:
            l = int(secret_len or 0)
            seed = int(self.owner.get_generator().integers(0, 2 ** 63))
            final_bits = two_universal_extract(bits, seed, l)
            final_key = bits_to_int(final_bits) if final_bits else None

            self.final_len_list.append(l)
            self.final_keys.append(final_key)

            msg = PAMessage(PAMsgType.PA_SEED, self.peer_name,
                            key_index=key_index, seed=seed, secret_len=l)
            self.owner.send_message(self.peer_node, msg)
            self._pop(final_key=final_key, secret_len=l, key_index=key_index)
        else:
            self._bob_buffer[key_index] = bits
            msg = self._bob_pending.pop(key_index, None)
            if msg is not None:
                self._bob_apply(key_index, msg)

    def _bob_apply(self, key_index: int, msg: PAMessage) -> None:
        bits = self._bob_buffer.pop(key_index)
        l = int(msg.kwargs["secret_len"])
        final_bits = two_universal_extract(bits, msg.kwargs["seed"], l)
        final_key = bits_to_int(final_bits) if final_bits else None
        self.final_len_list.append(l)
        self.final_keys.append(final_key)
        self._pop(final_key=final_key, secret_len=l, key_index=key_index)

    # ------------------------------------------------------------------
    def received_message(self, src: str, msg: PAMessage) -> bool:
        if msg.msg_type is PAMsgType.PA_SEED:            # Bob
            idx = msg.kwargs["key_index"]
            if idx in self._bob_buffer:
                self._bob_apply(idx, msg)
            else:
                self._bob_pending[idx] = msg
        return True

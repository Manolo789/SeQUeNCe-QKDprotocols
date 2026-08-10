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

Implementation note: the extractor is realised as a seeded BLAKE2b in
counter mode over the input string. This is a computational stand-in for a
Toeplitz two-universal hash; at the metric level the two are equivalent,
because the campaign statistics depend only on the OUTPUT LENGTH ``l``
(the composable epsilons are carried analytically by the entropy layer)
and on the equality of Alice's and Bob's outputs, which is guaranteed for
identical inputs by any deterministic function of (input, seed). A
GF(2)-Toeplitz implementation can be swapped in without touching the rest
of the stack.
"""

from __future__ import annotations

import hashlib
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..message import Message
from ..protocol import StackProtocol
from .error_correction import bits_to_int

if TYPE_CHECKING:
    from ..topology.node import Node


def two_universal_extract(bits: list, seed: int, out_len: int) -> list:
    """Compress ``bits`` to ``out_len`` bits with the seeded extractor.

    Args:
        bits (list[int]): reconciled key.
        seed (int): public extractor seed (chosen by Alice).
        out_len (int): output length l from the entropy-estimation layer.

    Returns:
        list[int]: final key bits (empty when ``out_len`` <= 0).
    """
    if out_len <= 0 or not bits:
        return []
    payload = bytes(bits)
    seed_bytes = int(seed).to_bytes(16, "big", signed=False)
    out = []
    counter = 0
    while len(out) < out_len:
        block = hashlib.blake2b(payload,
                                key=seed_bytes[:32],
                                salt=counter.to_bytes(8, "big")[:8],
                                digest_size=64).digest()
        for byte in block:
            for shift in range(7, -1, -1):
                out.append((byte >> shift) & 1)
                if len(out) == out_len:
                    return out
        counter += 1
    return out


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

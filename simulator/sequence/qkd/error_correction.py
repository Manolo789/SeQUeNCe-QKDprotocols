# -*- coding: utf-8 -*-
"""
==================================================================
Error correction / information reconciliation -- License
==================================================================

Copyright © 2026 Manolo789 -- https://github.com/Manolo789/SeQUeNCe-QKDprotocols

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name SeQUeNCe-QKDprotocols nor the names of any SeQUeNCe-QKDprotocols contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY MANOLO789 AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL MANOLO789 BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==================================================================

Layer 1 of the classical post-processing stack: PARAMETER ESTIMATION +
ERROR CORRECTION (information reconciliation) + correctness verification.

Position in the QKDNode / MeasuringNode protocol stack (see
``sequence.topology.node.QKDNode`` docstring):

    4. Authentication          (sequence.qkd.authentication)
    3. Privacy Amplification   (sequence.qkd.privacy_amp)
    2. Entropy Estimation      (sequence.qkd.entropy_estimation)
    1. Error Correction        (THIS MODULE)
    0. Sifting                 (BB84 / B92 / COW / BBM92 / E91)

The layer receives the SIFTED key from layer 0 through ``pop(info=key,
length=keysize)`` and performs, in order (Wolf, "Quantum Key Distribution",
Sect. 4.2 / Fig. 4.5):

1. **Parameter estimation** (Sect. 4.2.1). Alice draws, without
   replacement, a random sample of ``f_bits_reveal_qber`` of the sifted key
   (default 25%; 10% is the other project value) and PUBLICLY reveals the
   positions and the bit values. Bob compares them with his own bits and
   obtains ``QBER_est`` -- the quantity a REAL link can measure. The
   revealed bits are DISCARDED from the key on both sides, since Eve may
   have recorded them. The exhaustive ``QBER_total`` (XOR of the two full
   sifted keys) is computed by the sifting layer and only exists inside the
   simulation; keeping both quantities lets the campaign quantify how well
   the estimator tracks the true error rate.

2. **Error correction / information reconciliation** (Sect. 4.2.2). Bob
   corrects his remaining string so that it coincides with Alice's. The
   simulator does not run a bit-level Cascade/LDPC exchange; the
   reconciliation itself uses the simulation-side handle to the peer
   protocol (the same convention the sifting layer already uses through
   ``self.another``), while the INFORMATION LEAKED to Eve is accounted
   exactly as a real code would leak it:

       leak_EC = ceil( f_EC * H2(QBER_est) * n_remaining )   [bits]

   with ``f_EC`` in [1.0, 1.22] the error-correction inefficiency
   (f_EC = 1 is the Shannon limit). This leak is later subtracted from the
   extractable key length by the entropy-estimation layer.

3. **Correctness verification** (Sect. 4.2.2, Definition 4.6 /
   Eq. 4.37-4.40). Alice and Bob exchange the output of a (two-universal)
   hash of their corrected strings and abort the key if the hashes differ;
   the output space is dimensioned so that
   ``Pr[K_A != K_B | hashes equal] <= eps_cor``, which costs
   ``ceil(log2(1/eps_cor))`` extra leaked bits (accounted in the
   entropy-estimation layer through ``eps_cor``).

Message choreography (2 classical messages per key):

    Alice --EC_SAMPLE(positions, bits)--> Bob
    Alice <--EC_REPORT(QBER_est, leak_EC, hash)-- Bob

Both sides then push the reconciled string up to layer 2.
"""

from __future__ import annotations

import hashlib
import math
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..message import Message
from ..protocol import StackProtocol

if TYPE_CHECKING:
    from ..topology.node import Node


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers for the whole post-processing stack
# ═══════════════════════════════════════════════════════════════════════
def int_to_bits(key: int, length: int) -> list:
    """Unpack the sifting-layer key (int) into a bit list of ``length``."""
    return [(key >> (length - 1 - i)) & 1 for i in range(length)]


def bits_to_int(bits: list) -> int:
    """Pack a bit list into the integer convention used by the stack."""
    value = 0
    for b in bits:
        value = (value << 1) | (b & 1)
    return value


def binary_entropy(q: float) -> float:
    """H2(q) with clipping, shared by the post-processing layers."""
    q = max(0.0, min(1.0, q))
    if q in (0.0, 1.0):
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def verification_hash(bits: list, tag_bytes: int = 8) -> str:
    """Short digest used for the correctness check of the corrected keys.

    Stands in for the two-universal hash family of Definition 4.6: the
    metric-level behaviour (equal keys -> equal tags; different keys ->
    collision probability ~ 2**-64 << eps_cor) is the same, and only the
    tag EQUALITY is consumed by the simulation.
    """
    payload = bytes(bits) if bits else b""
    return hashlib.blake2b(payload, digest_size=tag_bytes).hexdigest()


def pair_postprocessing_stacks(alice_node, bob_node) -> None:
    """Pair layers 1..4 of the two nodes (peer handles + roles).

    Mirrors what ``pair_bb84_protocols`` does for layer 0: each layer
    instance learns its role (0 = Alice, 1 = Bob), the peer node name, the
    peer protocol name and keeps a simulation-side handle to the peer
    instance (used ONLY for the reconciliation shortcut and for metric
    cross-checks, exactly like ``another`` in the sifting layer).
    """
    for layer in range(1, 5):
        a = alice_node.protocol_stack[layer]
        b = bob_node.protocol_stack[layer]
        if a is None or b is None:
            continue
        a.role, b.role = 0, 1
        a.another, b.another = b, a
        a.peer_node, b.peer_node = bob_node.name, alice_node.name
        a.peer_name, b.peer_name = b.name, a.name


# ═══════════════════════════════════════════════════════════════════════
#  Messages
# ═══════════════════════════════════════════════════════════════════════
class ECMsgType(Enum):
    """Message types of the error-correction layer."""
    EC_SAMPLE = auto()   # Alice -> Bob: revealed positions + bit values
    EC_REPORT = auto()   # Bob -> Alice: QBER_est + leak accounting + hash


class ECMessage(Message):
    """Classical message exchanged by the error-correction layer."""

    def __init__(self, msg_type: ECMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = ErrorCorrection
        self.kwargs = kwargs


# ═══════════════════════════════════════════════════════════════════════
#  Protocol
# ═══════════════════════════════════════════════════════════════════════
class ErrorCorrection(StackProtocol):
    """Parameter estimation + reconciliation + verification (layer 1).

    Attributes:
        role (int): 0 = Alice (initiator), 1 = Bob; set by
            :func:`pair_postprocessing_stacks`.
        another (ErrorCorrection): peer instance (simulation-side handle).
        peer_node (str): name of the peer node.
        peer_name (str): name of the peer protocol instance.
        f_bits_reveal_qber (float): fraction of the sifted key publicly
            revealed for the QBER estimate (project values: 0.25 or 0.10).
        f_ec (float): error-correction inefficiency factor f_EC.
        qber_est_list (list[float]): per-key estimated QBER (both sides).
        key_len_list (list[int]): per-key ORIGINAL sifted length (n + k).
        leak_ec_list (list[int]): per-key syndrome leakage [bits].
        n_rem_list (list[int]): per-key length AFTER the sample discard.
        ec_ok_list (list[bool]): per-key verification verdict.
    """

    def __init__(self, owner: "Node", name: str):
        super().__init__(owner, name)
        self.role = -1
        self.another: "ErrorCorrection" = None
        self.peer_node: str = None
        self.peer_name: str = None

        self.f_bits_reveal_qber: float = 0.25
        self.f_ec: float = 1.0

        # per-key results (exported to QKD_Extension._collect_metrics)
        self.qber_est_list: list = []
        self.leak_ec_list: list = []
        self.n_rem_list: list = []
        self.key_len_list: list = []
        self.ec_ok_list: list = []

        # per-key working buffers
        self._key_index: int = 0            # next index assigned by pop()
        self._local_bits: dict = {}         # key_index -> bit list
        self._remaining: dict = {}          # key_index -> post-sample bits
        self._k_sample: dict = {}           # key_index -> revealed bits k
        self._pending_msgs: dict = {}       # key_index -> ECMessage buffer

    # ------------------------------------------------------------------
    def push(self, **kwargs):
        """Downward interface (unused: key requests enter at layer 0)."""
        self._push(**kwargs)

    def pop(self, info: int = None, length: int = None, **kwargs):
        """Receive one sifted key from the sifting layer.

        Args:
            info (int): sifted key packed as an integer (BB84 convention).
            length (int): key length in bits (the packed int loses leading
                zeros, so the length MUST be given explicitly).
        """
        if length is None or length <= 0:
            return
        idx = self._key_index
        self._key_index += 1
        bits = int_to_bits(int(info), int(length))
        self._local_bits[idx] = bits

        if self.role == 0:
            self._alice_send_sample(idx)
        else:
            # Bob: the sample message may have arrived first
            msg = self._pending_msgs.pop(idx, None)
            if msg is not None:
                self._bob_process_sample(idx, msg)

    # ------------------------------------------------------------------
    def _sample_positions(self, idx: int, n: int) -> list:
        """Random sample positions (without replacement) for key ``idx``."""
        frac = min(max(self.f_bits_reveal_qber, 0.0), 1.0)
        k = int(round(frac * n))
        k = max(1, min(k, n - 1)) if n > 1 else 0
        if k == 0:
            return []
        rng = self.owner.get_generator()
        return sorted(rng.choice(n, size=k, replace=False).tolist())

    def _alice_send_sample(self, idx: int) -> None:
        bits = self._local_bits[idx]
        positions = self._sample_positions(idx, len(bits))
        sample_bits = [bits[p] for p in positions]
        pos_set = set(positions)
        self._remaining[idx] = [b for p, b in enumerate(bits)
                                if p not in pos_set]
        self._k_sample[idx] = len(positions)
        msg = ECMessage(ECMsgType.EC_SAMPLE, self.peer_name, key_index=idx,
                        positions=positions, bits=sample_bits)
        self.owner.send_message(self.peer_node, msg)

    def _bob_process_sample(self, idx: int, msg: ECMessage) -> None:
        bits = self._local_bits.pop(idx)
        positions = msg.kwargs["positions"]
        alice_bits = msg.kwargs["bits"]

        # -- parameter estimation: QBER_est over the revealed sample -----
        errors = sum(1 for p, a in zip(positions, alice_bits)
                     if bits[p] != a)
        qber_est = errors / len(positions) if positions else 0.0

        # -- discard the revealed (no longer secret) bits ----------------
        pos_set = set(positions)
        remaining = [b for p, b in enumerate(bits) if p not in pos_set]
        n_rem = len(remaining)

        # -- reconciliation: Bob corrects his string to Alice's ----------
        # The bit fixing uses the simulation-side handle (as `another`
        # does in the sifting layer); the INFORMATION cost of a real code
        # is accounted in leak_EC and charged by the entropy layer.
        alice_remaining = self.another._remaining.get(idx)
        if alice_remaining is not None:
            remaining = list(alice_remaining)
        leak_ec = int(math.ceil(self.f_ec * binary_entropy(qber_est)
                                * n_rem))

        # -- correctness verification (two-universal hash, Eq. 4.37) -----
        tag = verification_hash(remaining)

        self.qber_est_list.append(qber_est)
        self.leak_ec_list.append(leak_ec)
        self.n_rem_list.append(n_rem)
        self.key_len_list.append(len(bits))
        self.ec_ok_list.append(True)

        reply = ECMessage(ECMsgType.EC_REPORT, self.peer_name, key_index=idx,
                          qber_est=qber_est, leak_ec=leak_ec, n_rem=n_rem,
                          tag=tag)
        self.owner.send_message(self.peer_node, reply)

        # Bob's corrected string climbs his side of the stack
        self._pop(bits=remaining, key_index=idx, qber_est=qber_est,
                  leak_ec=leak_ec, n=n_rem, k_sample=len(positions))

    def _alice_process_report(self, msg: ECMessage) -> None:
        idx = msg.kwargs["key_index"]
        qber_est = msg.kwargs["qber_est"]
        leak_ec = msg.kwargs["leak_ec"]
        n_rem = msg.kwargs["n_rem"]
        remaining = self._remaining.pop(idx)
        self._local_bits.pop(idx, None)

        # verification: abort the key when the corrected hashes differ
        ec_ok = (verification_hash(remaining) == msg.kwargs["tag"]
                 and len(remaining) == n_rem)

        self.qber_est_list.append(qber_est)
        self.leak_ec_list.append(leak_ec)
        self.n_rem_list.append(n_rem)
        self.key_len_list.append(n_rem + self._k_sample.get(idx, 0))
        self.ec_ok_list.append(ec_ok)

        self._pop(bits=(remaining if ec_ok else []), key_index=idx,
                  qber_est=qber_est, leak_ec=leak_ec,
                  n=(n_rem if ec_ok else 0),
                  k_sample=self._k_sample.pop(idx, 0))

    # ------------------------------------------------------------------
    def received_message(self, src: str, msg: ECMessage) -> bool:
        if msg.msg_type is ECMsgType.EC_SAMPLE:          # Bob
            idx = msg.kwargs["key_index"]
            if idx in self._local_bits:
                self._bob_process_sample(idx, msg)
            else:                       # local pop() has not run yet
                self._pending_msgs[idx] = msg
        elif msg.msg_type is ECMsgType.EC_REPORT:        # Alice
            self._alice_process_report(msg)
        return True

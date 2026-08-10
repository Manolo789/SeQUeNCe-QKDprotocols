# -*- coding: utf-8 -*-
"""
==================================================================
Authentication (post-processing layer 4) -- License
==================================================================

Copyright © 2026 Manolo789 -- https://github.com/Manolo789/SeQUeNCe-QKDprotocols

All rights reserved. (Same MIT-style terms as the rest of the extension;
see error_correction.py for the full text.)

==================================================================

Layer 4 (top) of the classical post-processing stack: AUTHENTICATION.

All classical post-processing messages must be authenticated, otherwise a
man-in-the-middle could impersonate Alice or Bob during sifting, parameter
estimation, error correction and privacy amplification. The layer models
Wegman-Carter authentication (Wegman & Carter, JCSS 22, 265 (1981)): each
authenticated message consumes ``tag_bits = ceil(log2(1/eps_auth))`` bits
of pre-shared key, which must be REPLENISHED from the freshly generated
key. The net secret key of the round is therefore

    l_net = max(0, l_PA - n_msgs * tag_bits),

with ``n_msgs`` the classical post-processing messages of the round
(EC sample, EC report, PA seed and the closing authentication tag:
n_msgs = 4 in this stack). ``R_sk`` is measured HERE, at the output of the
whole pipeline:

    R_sk = l_net / (qubits sent per train),

replacing the asymptotic Kržič Eq. (2.11) estimate used before the
post-processing was implemented.

Choreography (1 classical message per key):

    Alice <--AUTH_TAG(tag)-- Bob

Bob authenticates the round transcript (here condensed into the digest of
the final key and the key index) with the pre-shared key; Alice verifies
the tag and, only then, ACCEPTS the final key and records the per-key
metrics. The Alice instance also detects the end of the whole request
(sifting no longer working and no key left in flight) and stops the
simulation timeline, so runs do not keep replaying dark-count/background
events after the last key is delivered.
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

#: pre-shared secret used by the Wegman-Carter MAC of the simulation. In a
#: real deployment this is the (secret) authentication key; in the
#: simulation only tag EQUALITY and the KEY COST are consumed.
_PRESHARED_KEY = b"SeQUeNCe-QKD-preshared-authentication-key"

#: classical post-processing messages authenticated per generated key
#: (EC_SAMPLE, EC_REPORT, PA_SEED, AUTH_TAG).
N_AUTH_MESSAGES_PER_KEY = 4


def wegman_carter_tag(key_index: int, final_key: int, tag_bytes: int = 8) -> str:
    """Authentication tag of one round transcript (simulation stand-in)."""
    payload = (int(key_index).to_bytes(8, "big", signed=False)
               + (int(final_key) if final_key else 0).to_bytes(
                   max(1, ((int(final_key).bit_length() if final_key else 1)
                           + 7) // 8), "big", signed=False))
    return hashlib.blake2b(payload, key=_PRESHARED_KEY[:32],
                           digest_size=tag_bytes).hexdigest()


class AuthMsgType(Enum):
    """Message types of the authentication layer."""
    AUTH_TAG = auto()    # Bob -> Alice: MAC of the round transcript


class AuthMessage(Message):
    """Classical message exchanged by the authentication layer."""

    def __init__(self, msg_type: AuthMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = Authentication
        self.kwargs = kwargs


class Authentication(StackProtocol):
    """Wegman-Carter authentication + final-key delivery (layer 4).

    Attributes:
        role (int): 0 = Alice, 1 = Bob.
        eps_auth (float): MAC forgery probability; sets the per-message
            key cost tag_bits = ceil(log2(1/eps_auth)).
        final_key_lengths (list[int]): per-key NET secret length l_net
            (Alice; the campaign metric R_sk = l_net / qubits sent).
        auth_ok_list (list[bool]): per-key tag verification verdict.
        key_times (list[int]): simulation time [ps] each key was accepted.
        latency (float): time to the FIRST authenticated key [s].
    """

    def __init__(self, owner: "Node", name: str):
        super().__init__(owner, name)
        self.role = -1
        self.another: "Authentication" = None
        self.peer_node: str = None
        self.peer_name: str = None

        self.eps_auth: float = 1e-10
        self.stop_timeline_when_done: bool = True

        self.final_key_lengths: list = []
        self.auth_ok_list: list = []
        self.key_times: list = []
        self.latency: float = 0.0

        self._alice_buffer: dict = {}   # key_index -> (final_key, l)
        self._alice_pending: dict = {}  # key_index -> AUTH_TAG message

    # ------------------------------------------------------------------
    @property
    def tag_bits(self) -> int:
        """Pre-shared key bits consumed per authenticated message."""
        return int(math.ceil(math.log2(1.0 / self.eps_auth)))

    def _sifting(self):
        """Handle to the layer-0 protocol of this node's stack."""
        proto = self
        while proto.lower_protocols:
            proto = proto.lower_protocols[0]
        return proto

    def _net_length(self, l_pa: int) -> int:
        """Net secret length after replenishing the authentication key."""
        if l_pa <= 0:
            return 0
        return max(0, int(l_pa) - N_AUTH_MESSAGES_PER_KEY * self.tag_bits)

    # ------------------------------------------------------------------
    def push(self, **kwargs):
        self._push(**kwargs)

    def pop(self, final_key: int = None, secret_len: int = 0,
            key_index: int = 0, **kwargs):
        """Receive one final key from the privacy-amplification layer."""
        if self.role == 1:
            # Bob closes the round: authenticate the transcript
            tag = wegman_carter_tag(key_index, final_key)
            msg = AuthMessage(AuthMsgType.AUTH_TAG, self.peer_name,
                              key_index=key_index, tag=tag)
            self.owner.send_message(self.peer_node, msg)
            return

        # Alice: wait for Bob's tag before accepting the key
        self._alice_buffer[key_index] = (final_key, int(secret_len or 0))
        msg = self._alice_pending.pop(key_index, None)
        if msg is not None:
            self._alice_verify(key_index, msg)

    def _alice_verify(self, key_index: int, msg: AuthMessage) -> None:
        final_key, l_pa = self._alice_buffer.pop(key_index)
        auth_ok = (msg.kwargs["tag"] == wegman_carter_tag(key_index,
                                                          final_key))
        l_net = self._net_length(l_pa) if auth_ok else 0

        now = self.owner.timeline.now()
        self.final_key_lengths.append(l_net)
        self.auth_ok_list.append(auth_ok)
        self.key_times.append(now)
        if self.latency == 0.0 and l_net >= 0:
            self.latency = now * 1e-12

        self._maybe_stop_timeline()

    def _maybe_stop_timeline(self) -> None:
        """Stop the timeline when the whole request has been delivered.

        Without this, high dark-count / bright-sky runs keep replaying the
        detector noise chains until the full timeline horizon, which is
        both slow and unbounded in memory (the cause of the truncated
        ``dark_count`` sweep of the 2026-08-08 campaign log).
        """
        if not self.stop_timeline_when_done or self.role != 0:
            return
        sift = self._sifting()
        produced = len(getattr(sift, "error_rates", []))
        if (not getattr(sift, "working", False)
                and not self._alice_buffer
                and len(self.final_key_lengths) >= produced):
            self.owner.timeline.stop()

    # ------------------------------------------------------------------
    def received_message(self, src: str, msg: AuthMessage) -> bool:
        if msg.msg_type is AuthMsgType.AUTH_TAG:         # Alice
            idx = msg.kwargs["key_index"]
            if idx in self._alice_buffer:
                self._alice_verify(idx, msg)
            else:
                self._alice_pending[idx] = msg
        return True

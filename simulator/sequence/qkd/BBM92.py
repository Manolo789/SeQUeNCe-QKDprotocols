# -*- coding: utf-8 -*-
"""
==================================================================
Extension of the BBM92 protocol to the SeQUeNCe simulator -- License
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

Definition of the BBM92 entanglement-based QKD protocol.

BBM92 (Bennett-Brassard-Mermin 1992) is the entanglement-based version of
BB84. An untrusted central source (Charlie) distributes Bell pairs; Alice
and Bob each measure their photon in one of TWO conjugate bases
(Z = 0 deg rectilinear, X = 45 deg diagonal). Classical post-processing:

  1. sifting: keep only the rounds where BOTH used the SAME basis;
  2. a random sample of the sifted key is revealed to estimate the QBER
     and then DISCARDED from the key on both sides (Kržič Fig. 2.5);
  3. security verdict via the asymptotic key fraction (Kržič Eq. 2.11):
     an intercept-resend attack raises the QBER to ~25%, well above the
     abort threshold (E_bound = 11% for f_EC=1 ... 9.4% for f_EC=1.22).

For the |Phi+> source, same-basis outcomes are equal; for the singlet
|Psi-> (the dissertation source, Eq. 3.1) they are anti-correlated and
Bob flips his bits (``anti_correlated=True``, the default).

The shared physical layer (Bell-pair source, measuring nodes, messages)
lives in :mod:`sequence.qkd.entanglement`; the network-facing node classes
(``EntanglementSourceNode``, ``MeasuringNode``) live in
:mod:`sequence.topology.node`, just like ``QKDNode`` does for BB84/B92/COW.
"""

from __future__ import annotations

from .entanglement import (
    BaseEntanglementQKD, EntQKDMessage, EntQKDMsgType,
    pair_entanglement_protocols,
)


def pair_bbm92_protocols(sender: "BBM92", receiver: "BBM92",
                         anti_correlated: bool = None) -> None:
    """Function to pair BBM92 protocol instances.

    Args:
        sender (BBM92): protocol instance starting the classical phase (Alice).
        receiver (BBM92): protocol instance performing the sifting (Bob).
        anti_correlated (bool): True for the |Psi-> source (optional override).
    """
    pair_entanglement_protocols(sender, receiver, anti_correlated)


class BBM92(BaseEntanglementQKD):
    """Implementation of the BBM92 protocol (see module docstring).

    Sifting keeps same-basis rounds; a public sample estimates the QBER and
    is discarded from the key on both sides. Metrics (on Alice's side after
    each train): ``sifted_len``, ``sample_len``, ``qber``; the per-train
    sample QBER is also accumulated in ``sampled_qbers`` as a diagnostic.

    Key generation is keysize-oriented and lives in the base class: the
    sifted bits of successive trains accumulate until ``keysize`` bits are
    available (see :class:`~sequence.qkd.entanglement.BaseEntanglementQKD`).
    """

    #: Two conjugate bases: Z = 0 deg (rectilinear), X = 45 deg (diagonal).
    ANGLES_ALICE = [0.0, 45.0]
    ANGLES_BOB = [0.0, 45.0]

    # ----- Bob: sifting -----------------------------------------------------
    # Parameter estimation (QBER sample reveal + discard) was MOVED to the
    # shared post-processing stack (sequence.qkd.error_correction), so that
    # BOTH protocol families estimate and discard the SAME fraction of the
    # sifted key. The old per-train sample here made BBM92 lose 25% of its
    # sifted bits BEFORE key accumulation while BB84/B92/COW lost none,
    # biasing every R_s / R_sk comparison between the families -- and it
    # would have DOUBLE-discarded once the stack ran its own estimation.
    def _bob_sift(self, common: list, alice_settings: dict):
        # keep rounds where both sides used the SAME basis
        sift_rounds = [i for i in common
                       if alice_settings[i] == self.owner.records[i][0]]
        self.key_rounds = sift_rounds
        # bits of THIS train; the keysize-oriented driver in the base class
        # accumulates them into `key_bits` until a full key is available.
        self.train_key_bits = [self._bit(i) for i in sift_rounds]
        self.metrics = {"sifted_len": len(sift_rounds)}

        reply = EntQKDMessage(EntQKDMsgType.SIFT_ANNOUNCE, self.peer_proto,
                              key_rounds=sift_rounds,
                              sifted_len=len(sift_rounds))
        self.owner.send_message(self.peer_node, reply)

    # ----- Alice: per-train metric (key already adopted by the base class) ---
    def _alice_finish(self, msg: EntQKDMessage):
        self.metrics = {"sifted_len": msg.kwargs["sifted_len"]}

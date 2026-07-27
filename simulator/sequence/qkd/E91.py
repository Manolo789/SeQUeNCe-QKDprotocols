# -*- coding: utf-8 -*-
"""
==================================================================
Extension of the E91 protocol to the SeQUeNCe simulator -- License
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

Definition of the E91 (Ekert 1991) entanglement-based QKD protocol.

An untrusted central source (Charlie) distributes Bell pairs; Alice and
Bob each measure their photon with one of THREE analyser angles:

    Alice: a0 = 0 deg,    a1 = 22.5 deg,  a2 = 45 deg
    Bob  : b0 = 22.5 deg, b1 = 45 deg,    b2 = 67.5 deg

Classical post-processing:

  1. key rounds: combinations with physically equal angles
     ((a1, b0) and (a2, b1)) give perfect (anti-)correlation and produce
     the key -- for the singlet |Psi-> Bob flips his bits
     (``anti_correlated=True``, the default);
  2. test rounds: Alice in {a0, a2} x Bob in {b0, b2} feed the CHSH
     Bell-inequality test
         S = E(a0,b0) - E(a0,b2) + E(a2,b0) + E(a2,b2),
     with E(a,b) = +-cos 2(a-b) -> |S| = 2*sqrt(2) ~ 2.83 for an intact
     Bell state (violating the classical bound |S| <= 2). An
     intercept-resend attack destroys the entanglement and drives
     |S| below 2, revealing the eavesdropper. Bob's test outcomes are
     public information (never part of the key).

The shared physical layer (Bell-pair source, messages, base protocol)
lives in :mod:`sequence.qkd.entanglement`; the network-facing node classes
(``EntanglementSourceNode``, ``MeasuringNode``) live in
:mod:`sequence.topology.node`, just like ``QKDNode`` does for BB84/B92/COW.
"""

from __future__ import annotations

from .entanglement import (
    BaseEntanglementQKD, EntQKDMessage, EntQKDMsgType,
    pair_entanglement_protocols,
)


def pair_e91_protocols(sender: "E91", receiver: "E91",
                       anti_correlated: bool = None) -> None:
    """Function to pair E91 protocol instances.

    Args:
        sender (E91): protocol instance starting the classical phase (Alice).
        receiver (E91): protocol instance performing the sifting (Bob).
        anti_correlated (bool): True for the |Psi-> source (optional override).
    """
    pair_entanglement_protocols(sender, receiver, anti_correlated)


class E91(BaseEntanglementQKD):
    """Implementation of the E91 protocol (see module docstring).

    Metrics (on Alice's side after the exchange): ``chsh_S`` (signed CHSH
    value; the security verdict uses |S| > 2 for both Bell states) and
    ``key_rounds_len``.
    """

    #: three analyser angles per party [degrees].
    ANGLES_ALICE = [0.0, 22.5, 45.0]
    ANGLES_BOB = [22.5, 45.0, 67.5]

    #: (alice_setting, bob_setting) pairs with equal physical angle -> key.
    KEY_PAIRS = {(1, 0), (2, 1)}
    #: CHSH settings: Alice {0 deg, 45 deg} x Bob {22.5 deg, 67.5 deg}.
    CHSH_A = (0, 2)
    CHSH_B = (0, 2)

    # ----- Bob: split rounds into key / CHSH-test ---------------------------
    def _bob_sift(self, common: list, alice_settings: dict):
        # key rounds: physically equal analyser angles
        key_rounds = [i for i in common
                      if (alice_settings[i],
                          self.owner.records[i][0]) in self.KEY_PAIRS]
        self.key_rounds = key_rounds
        self.key = [self._bit(i) for i in key_rounds]
        self.metrics = {"key_rounds_len": len(key_rounds)}

        # test rounds (public): Bob reveals setting + outcome for the CHSH
        test_rounds = [i for i in common
                       if alice_settings[i] in self.CHSH_A
                       and self.owner.records[i][0] in self.CHSH_B]
        test_data = [(i, self.owner.records[i][0],
                      self.owner.records[i][1]) for i in test_rounds]

        reply = EntQKDMessage(EntQKDMsgType.SIFT_ANNOUNCE, self.peer_proto,
                              key_rounds=key_rounds, test_data=test_data)
        self.owner.send_message(self.peer_node, reply)

    # ----- Alice: CHSH metric (key already adopted by the base class) --------
    def _alice_finish(self, msg: EntQKDMessage):
        test_data = msg.kwargs["test_data"]
        self.metrics = {"chsh_S": self._chsh(test_data),
                        "key_rounds_len": len(self.key_rounds)}

    # ----- CHSH ---------------------------------------------------------------
    def _chsh(self, test_data_bob):
        """S = E(a0,b0) - E(a0,b2) + E(a2,b0) + E(a2,b2).

        Uses Alice's local outcomes and Bob's revealed (public) test data.
        For |Phi+> S -> +2*sqrt(2); for |Psi-> S -> -2*sqrt(2). The
        security verdict uses |S| > 2 in both cases.
        """
        acc = {(a, b): [0, 0] for a in self.CHSH_A for b in self.CHSH_B}
        for i, b_setting, b_outcome in test_data_bob:
            if i not in self.owner.records:
                continue
            a_setting, a_outcome = self.owner.records[i]
            if a_setting not in self.CHSH_A or b_setting not in self.CHSH_B:
                continue
            sa = 1 - 2 * a_outcome   # 0 -> +1, 1 -> -1
            sb = 1 - 2 * b_outcome
            acc[(a_setting, b_setting)][0] += sa * sb
            acc[(a_setting, b_setting)][1] += 1

        def E(a, b):
            s, n = acc[(a, b)]
            return s / n if n else 0.0

        a0, a2 = self.CHSH_A
        b0, b2 = self.CHSH_B
        S = E(a0, b0) - E(a0, b2) + E(a2, b0) + E(a2, b2)
        # per-combination counts, kept for diagnostics
        self._chsh_counts = {k: acc[k][1] for k in acc}
        return S

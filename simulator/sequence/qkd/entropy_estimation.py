# -*- coding: utf-8 -*-
"""
==================================================================
Entropy estimation (post-processing layer 2) -- License
==================================================================

Copyright © 2026 Manolo789 -- https://github.com/Manolo789/SeQUeNCe-QKDprotocols

All rights reserved. (Same MIT-style terms as the rest of the extension;
see error_correction.py for the full text.)

==================================================================

Layer 2 of the classical post-processing stack: ENTROPY ESTIMATION.

Bounds the smooth min-entropy of Alice's reconciled string conditioned on
Eve's quantum side information, and converts it into the number of secret
bits ``l`` that the privacy-amplification layer may extract (Wolf,
"Quantum Key Distribution", Sect. 4.2.3, Lemma 4.9 / Eq. 4.44):

    l = floor( n * (1 - I_E(Q_bound)) - leak_EC
               - log2(2 / eps_cor) - 2*log2(1 / (2*eps_pa)) )

where

* ``n`` is the reconciled key length (after the parameter-estimation
  sample discard);
* ``leak_EC`` is the syndrome leakage accounted by the error-correction
  layer (f_EC * H2(QBER_est) * n bits);
* ``Q_bound = QBER_est + gamma`` inflates the estimate with the
  finite-sampling penalty of Serfling's inequality (Theorem 4.5 /
  Eq. 4.35): the probability that the error rate of the unrevealed bits
  exceeds the sampled one by more than

      gamma = sqrt( (k + 1) * N * ln(1/eps_PE) / (2 * k^2 * n) )

  is at most ``eps_PE`` (k = revealed bits, N = n + k);
* ``I_E`` is the protocol-specific bound on Eve's information per bit:

    - BB84 / B92 / BBM92 (mode "h2"):  I_E = H2(Q_bound);
    - COW (mode "cow"): the security witness is the monitoring-line
      visibility V (Stucki et al., APL 87, 194108 (2005)):
          I_E = mu*(1 - t) + (1 - V) * (1 + e^{-mu t}) / (2 e^{-mu t}),
      with t the channel transmission and mu the mean photon number;
    - E91 (mode "e91"): Eve's information is bounded by the CHSH
      parameter, I_E = chi(S) (see :meth:`_eve_information`). If
      |S| <= 2 the key is ABORTED (l = 0); otherwise the standard
      entanglement-based bound I_E = H2(Q_bound) is applied (the
      device-independent bound based on S is listed as an open decision in
      the campaign notes).

The abort threshold is implicit: when ``1 - I_E - leak_EC/n`` becomes
non-positive, ``l = 0`` and the key is discarded, reproducing the
parameter-estimation abort of Fig. 4.5.

Only Alice's instance evaluates the bound (she owns the per-key COW
visibility and E91 CHSH series); the resulting ``l`` is transported to Bob
by the privacy-amplification layer together with the extractor seed, so
both sides always compress to the same length.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..protocol import StackProtocol
from .error_correction import binary_entropy

if TYPE_CHECKING:
    from ..topology.node import Node


class EntropyEstimation(StackProtocol):
    """Smooth min-entropy bound of the reconciled key (layer 2).

    Attributes:
        role (int): 0 = Alice, 1 = Bob (paired by
            ``pair_postprocessing_stacks``).
        mode (str): "h2" (BB84/B92/BBM92), "cow" or "e91".
        f_ec (float): error-correction inefficiency (documentation only;
            the actual leak arrives pre-computed from layer 1).
        eps_pe / eps_cor / eps_pa (float): failure probabilities of the
            parameter estimation, correctness check and privacy
            amplification (composable security parameters).
        cow_mu (float): COW mean photon number (mode "cow").
        channel_transmission (float): end-to-end transmission t (mode "cow").
        enforce_bell_violation (bool): abort E91 keys with |S| <= 2.
        secret_len_list (list[int]): per-key extractable length l (Alice).
        qber_bound_list (list[float]): per-key Q_bound = Q_est + gamma.
    """

    def __init__(self, owner: "Node", name: str):
        super().__init__(owner, name)
        self.role = -1
        self.another: "EntropyEstimation" = None
        self.peer_node: str = None
        self.peer_name: str = None

        self.mode: str = "h2"
        self.f_ec: float = 1.0
        self.eps_pe: float = 1e-10
        self.eps_cor: float = 1e-15
        self.eps_pa: float = 1e-10

        # protocol-specific context (set by run_qkd_simulation)
        self.cow_mu: float = None
        self.channel_transmission: float = None
        self.enforce_bell_violation: bool = True

        # per-key results (Alice)
        self.secret_len_list: list = []
        self.qber_bound_list: list = []

    # ------------------------------------------------------------------
    def _sifting(self):
        """Handle to the layer-0 protocol of this node's stack."""
        proto = self
        while proto.lower_protocols:
            proto = proto.lower_protocols[0]
        return proto

    def _serfling_gamma(self, n: int, k: int) -> float:
        """Finite-sampling penalty of Serfling's bound (Eq. 4.35).

        Args:
            n (int): unrevealed (key) bits.
            k (int): revealed (sample) bits.

        Returns:
            float: gamma such that Pr[Lambda_n >= Lambda_k + gamma] <=
            eps_pe (conditioned on passing the threshold check).
        """
        if n <= 0 or k <= 0:
            return 0.5
        N = n + k
        return math.sqrt((k + 1) * N * math.log(1.0 / self.eps_pe)
                         / (2.0 * k * k * n))

    def _eve_information(self, q_bound: float, key_index: int) -> float:
        """Protocol-specific bound I_E on Eve's information per key bit.

        * mode "h2" (BB84/B92/BBM92): the standard bound
          I_E = h2(Q_bound);
        * mode "cow": the monitoring-line visibility bound (photon-number
          splitting fraction + interference term);
        * mode "e91": the device-independent-style CHSH bound of the
          Devetak-Winter rate for CHSH-based protocols (Acin et al., PRL
          98, 230501, 2007): Eve's information is limited by the Bell
          parameter alone,

              I_E = chi(S) = h2((1 + sqrt((S/2)^2 - 1)) / 2),

          with S the mean CHSH value of the run so far. The bound
          penalises weak violations even at low QBER (chi -> 1 as
          |S| -> 2, so l -> 0 continuously) and vanishes at the Tsirelson
          limit |S| = 2*sqrt(2). The QBER still enters the secret length
          through the error-correction leakage, reproducing
          r = 1 - h2(Q) - chi(S). No finite-size correction is applied to
          S itself (the Serfling penalty covers the QBER estimate only);
          this simplification is documented in the project report.
        """
        if self.mode == "cow":
            sift = self._sifting()
            vis = getattr(sift, "visibility", None)
            v = vis[key_index] if vis and key_index < len(vis) else 0.0
            if not (v == v):            # NaN -> conservative worst case
                v = 0.0
            mu = self.cow_mu if self.cow_mu is not None else 0.5
            t = (self.channel_transmission
                 if self.channel_transmission is not None else 1.0)
            t = min(max(t, 1e-12), 1.0)
            return (mu * (1.0 - t)
                    + (1.0 - v) * (1.0 + math.exp(-mu * t))
                    / (2.0 * math.exp(-mu * t)))
        if self.mode == "e91":
            s_mean = self._chsh_mean()
            s_abs = abs(s_mean) if s_mean == s_mean else 0.0
            if s_abs <= 2.0:
                return 1.0              # no violation: nothing is secret
            ratio = min(s_abs / 2.0, math.sqrt(2.0))
            return binary_entropy((1.0 + math.sqrt(ratio * ratio - 1.0))
                                  / 2.0)
        # "h2": the standard BB84 bound
        return binary_entropy(q_bound)

    def _chsh_mean(self) -> float:
        """Mean CHSH parameter over the values accumulated so far."""
        sift = self._sifting()
        values = [s for s in getattr(sift, "chsh_values", []) if s == s]
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def _bell_check_passed(self) -> bool:
        """E91 verdict: |S| > 2 over the CHSH values accumulated so far."""
        s_mean = self._chsh_mean()
        return s_mean == s_mean and abs(s_mean) > 2.0

    def secret_length(self, n: int, k: int, qber_est: float,
                      leak_ec: int, key_index: int) -> tuple:
        """Extractable secret length of one reconciled key.

        Args:
            n (int): reconciled key length [bits].
            k (int): revealed sample length [bits].
            qber_est (float): estimated QBER of the sample.
            leak_ec (int): error-correction leakage [bits].
            key_index (int): index of the key (aligns V / S series).

        Returns:
            tuple: (l, q_bound) with l >= 0 the number of secret bits.
        """
        if n <= 0:
            return 0, 0.5
        if (self.mode == "e91" and self.enforce_bell_violation
                and not self._bell_check_passed()):
            return 0, min(0.5, qber_est)

        gamma = self._serfling_gamma(n, k)
        q_bound = min(0.5, qber_est + gamma)
        i_e = self._eve_information(q_bound, key_index)

        delta = (math.log2(2.0 / self.eps_cor)
                 + 2.0 * math.log2(1.0 / (2.0 * self.eps_pa)))
        l = math.floor(n * (1.0 - i_e) - leak_ec - delta)
        return max(0, int(l)), q_bound

    # ------------------------------------------------------------------
    def push(self, **kwargs):
        self._push(**kwargs)

    def pop(self, bits: list = None, key_index: int = 0,
            qber_est: float = 0.0, leak_ec: int = 0, n: int = 0,
            k_sample: int = 0, **kwargs):
        """Receive one reconciled key from the error-correction layer.

        ``k_sample`` (the number of publicly revealed bits) arrives
        explicitly from the error-correction layer, so the Serfling
        penalty never depends on transient sifting-layer state.
        """
        if self.role == 0:
            l, q_bound = self.secret_length(n, int(k_sample), qber_est,
                                            leak_ec, key_index)
            self.secret_len_list.append(l)
            self.qber_bound_list.append(q_bound)
            self._pop(bits=bits, key_index=key_index, secret_len=l)
        else:
            # Bob compresses to the length Alice announces with the seed
            self._pop(bits=bits, key_index=key_index, secret_len=None)

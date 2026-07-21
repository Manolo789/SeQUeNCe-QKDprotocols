# -*- coding: utf-8 -*-
"""
==================================================================
Shared infrastructure for entanglement-based QKD -- License
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

"""

"""Shared infrastructure for entanglement-based QKD (BBM92 / E91).

Both protocols share EXACTLY the same physical layer:

    +---------+     photon 0      +---------+     photon 1     +---------+
    |  Alice  | <---- q.ch. ----- | Charlie | ----- q.ch. ---> |   Bob   |
    | (meas.) |                   | (EPS)   |                  | (meas.) |
    +---------+                   +---------+                  +---------+
         ^                        Bell pair                         ^
         +-------------- classical channel (sifting) ---------------+

Charlie hosts an entangled pair source (EPS, untrusted relay -- Kržič
dissertation §2.1.2). Alice and Bob measure in randomly chosen bases; the
difference between BBM92 and E91 is ONLY the basis set and the classical
post-processing (basis reconciliation + QBER sampling vs. CHSH test).
Those protocol-specific parts live in :mod:`sequence.qkd.BBM92` and
:mod:`sequence.qkd.E91`; everything they share lives here.

Design notes for THIS fork (QC-in-FS):
  * The channel-level ``polarization_fidelity`` MUST be kept at 1.0:
    upstream ``FreeQuantumState.random_noise`` has an open
    "TODO: rewrite for entangled states" and corrupts the joint state
    (ValueError in ``Photon.measure``). A phenomenological per-photon
    depolarization is available in ``topology.node.MeasuringNode``
    instead (same average effect: (1-f)/2 of QBER per photon).
  * Free-space loss enters through the fork's ``QuantumChannel(loss=...)``
    override (computed by ``QCLoss.loss.channel_FSO_loss``); the standard
    attenuation formula is the fallback.
  * Daylight background enters as a per-coincidence-window noise
    probability derived from ``QCLoss.sky_radiance.n_background``
    (photons per detection gate), see
    ``topology.node.MeasuringNode.apply_noise_counts``.
  * Eve: reuse the fork's generic ``sequence.topology.node.EveNode``
    (typically via ``EveQuantumChannel``). It measures in a random Z/X
    basis and forwards the SAME photon object, so the collapse propagates
    to Bob's partner photon and ``round_index`` is preserved.

Asymptotic secure key rate (Kržič Eq. (2.11)):
    R = C_sift * [1 - f_EC * H2(E) - H2(E)],
with f_EC in [1.0, 1.22] (E_bound = 11% ... 9.4%).
"""

from __future__ import annotations

from enum import Enum, auto
from math import cos, sin, sqrt
from typing import TYPE_CHECKING

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..protocol import Protocol
from ..message import Message
from ..components.photon import Photon
from ..utils.encoding import polarization

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node, MeasuringNode


# ===========================================================================
# Quantum utilities
# ===========================================================================

#: Bell states in the 2-qubit HV (x) HV space.
PHI_PLUS = (complex(sqrt(0.5)), complex(0), complex(0), complex(sqrt(0.5)))
#: Singlet |Psi-> = (|HV> - |VH>)/sqrt(2): the state produced by the real
#: source of the Kržič dissertation (Eq. 3.1). Anti-correlated in ALL bases.
PSI_MINUS = (complex(0), complex(sqrt(0.5)), complex(-sqrt(0.5)), complex(0))

BELL_STATES = {"phi_plus": PHI_PLUS, "psi_minus": PSI_MINUS}


def basis_from_angle(theta_rad: float):
    """Polarization measurement basis for an analyser at angle ``theta``.

    Format expected by ``Photon.measure``:
        outcome 0 -> |theta>       = ( cos t,  sin t)
        outcome 1 -> |theta + 90>  = (-sin t,  cos t)
    """
    return ((complex(cos(theta_rad)), complex(sin(theta_rad))),
            (complex(-sin(theta_rad)), complex(cos(theta_rad))))


# ===========================================================================
# Classical messages
# ===========================================================================

class EntQKDMsgType(Enum):
    """Message types for the entanglement-based QKD classical phase."""
    BASIS_ANNOUNCE = auto()   # Alice -> Bob: detected rounds + basis settings
    SIFT_ANNOUNCE = auto()    # Bob -> Alice: key rounds / QBER sample / CHSH data


class EntQKDMessage(Message):
    """Message used by the entanglement-based QKD protocols (BBM92 / E91)."""

    def __init__(self, msg_type: EntQKDMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = BaseEntanglementQKD
        self.kwargs = kwargs


# ===========================================================================
# Hardware: entangled pair source (hosted by the central node "Charlie")
# ===========================================================================

class BellPairSource(Entity):
    """Source of polarization-entangled photon pairs.

    Each round creates two real SeQUeNCe photons, entangles them
    (``combine_state``) and pins the joint state to the chosen Bell state
    (``set_state``) -- the same mechanics used by the native ``SPDCSource``.
    If ``mean_photon_num`` is given, the number of pairs per round follows
    k ~ Poisson(mu) (multi-pair events produce accidental coincidences,
    Kržič §2.1.4); otherwise exactly one pair per round is emitted.

    ``photon.round_index`` is the coincidence tag used to pair detections
    between Alice and Bob (stand-in for the timestamp + coincidence-window
    machinery of a real system).

    Attributes:
        owner_node (Node): node hosting the source (Charlie).
        dst_alice (str): name of the next node on Alice's arm.
        dst_bob (str): name of the next node on Bob's arm.
        num_rounds (int): number of emission rounds.
        frequency (float): pair generation rate [Hz].
        wavelength_nm (float): photon wavelength [nm] (SeQUeNCe convention).
        state (tuple): 4-amplitude Bell state.
        mean_photon_num (float | None): Poisson mean; None => 1 pair/round.
        emitted (int): number of pairs emitted so far.
    """

    def __init__(self, name, timeline: "Timeline", owner_node: "Node",
                 dst_alice: str, dst_bob: str, num_rounds: int,
                 frequency: float = 1e6, wavelength_nm: float = 810,
                 bell_state: str = "psi_minus", mean_photon_num=None):
        super().__init__(name, timeline)
        self.owner_node = owner_node
        self.dst_alice = dst_alice
        self.dst_bob = dst_bob
        self.num_rounds = num_rounds
        self.frequency = frequency
        self.wavelength_nm = wavelength_nm
        self.state = BELL_STATES[bell_state]
        self.mean_photon_num = mean_photon_num
        self.emitted = 0

    def init(self):
        pass

    def start(self, start_time: int):
        """Schedule the emission of all rounds starting at ``start_time``."""
        period = int(round(1e12 / self.frequency))  # ps
        for i in range(self.num_rounds):
            process = Process(self, "emit_round", [i])
            self.timeline.schedule(Event(start_time + i * period, process))

    def emit_round(self, i: int):
        """Emit the pair(s) of round ``i``."""
        if self.mean_photon_num is None:
            n_pairs = 1
        else:
            n_pairs = int(self.owner_node.get_generator().poisson(
                self.mean_photon_num))
        for _ in range(n_pairs):
            self._emit_pair(i)

    def _emit_pair(self, i: int):
        """Create ONE Bell pair and distribute it to the two arms."""
        p_alice = Photon(f"pA_{i}", self.timeline,
                         wavelength=self.wavelength_nm,
                         location=self.owner_node,
                         encoding_type=polarization)
        p_bob = Photon(f"pB_{i}", self.timeline,
                       wavelength=self.wavelength_nm,
                       location=self.owner_node,
                       encoding_type=polarization)
        p_alice.combine_state(p_bob)
        p_alice.set_state(self.state)
        p_alice.round_index = i
        p_bob.round_index = i
        self.owner_node.send_qubit(self.dst_alice, p_alice)
        self.owner_node.send_qubit(self.dst_bob, p_bob)
        self.emitted += 1


# ===========================================================================
# Application: base class for entanglement-based QKD protocols
# ===========================================================================

class BaseEntanglementQKD(Protocol):
    """Classical layer common to BBM92 and E91.

    Alice announces her bases and detected rounds (BASIS_ANNOUNCE). Bob
    sifts and replies (SIFT_ANNOUNCE); the subclass defines what Bob
    computes (:meth:`_bob_sift`) and how Alice finishes
    (:meth:`_alice_finish`).

    Mirrors the conventions of BB84/B92/COW in this repository:
      * ``role`` is -1 until :func:`~sequence.qkd.BBM92.pair_bbm92_protocols`
        (or the E91 analog) is called: 0 = Alice (initiator), 1 = Bob.
      * The measuring hardware lives in the owner node
        (``topology.node.MeasuringNode``), just like the light source and
        detectors live in ``QKDNode`` for the prepare-and-measure protocols.

    If ``anti_correlated=True`` (state |Psi->) Bob flips his key/sample
    bits so that both keys coincide (Kržič Eq. 3.1).

    Class attributes (overridden by subclasses):
        ANGLES_ALICE (list[float]): analyser angles for Alice [degrees].
        ANGLES_BOB (list[float]): analyser angles for Bob [degrees].

    Attributes:
        owner (MeasuringNode): node the protocol is attached to.
        role (int): -1 unpaired, 0 Alice, 1 Bob.
        peer_node (str): name of the peer node.
        peer_proto (str): name of the peer protocol instance.
        anti_correlated (bool): True for the |Psi-> source.
        key (list[int]): final (sifted, post-discard) key bits.
        key_rounds (list[int]): round indices contributing to the key.
        metrics (dict): protocol metrics (qber / chsh_S / sifted_len ...).
    """

    #: analyser angles in degrees; subclasses must override.
    ANGLES_ALICE: list = []
    ANGLES_BOB: list = []

    def __init__(self, owner: "MeasuringNode", name: str,
                 peer_node_name: str = None, peer_protocol_name: str = None,
                 anti_correlated: bool = True):
        super().__init__(owner, name)
        self.role = -1                          # set by the pair_* function
        self.peer_node = peer_node_name
        self.peer_proto = peer_protocol_name
        self.anti_correlated = anti_correlated

        self.key: list = []
        self.key_rounds: list = []
        self.metrics: dict = {}

    def init(self):
        pass

    # ------------------------------------------------------------------
    @classmethod
    def angles_for_role(cls, role) -> list:
        """Analyser angles [degrees] for ``role`` (0/'alice' or 1/'bob')."""
        if role in (0, "alice"):
            return cls.ANGLES_ALICE
        if role in (1, "bob"):
            return cls.ANGLES_BOB
        raise ValueError(f"invalid role {role!r} (expected 0/1/'alice'/'bob')")

    def _bit(self, i: int) -> int:
        """Key bit of round ``i`` (Bob flips it for anti-correlated states)."""
        b = self.owner.records[i][1]
        if self.anti_correlated and self.role == 1:
            b ^= 1
        return b

    # ----- Alice starts the classical phase --------------------------------
    def announce_bases(self):
        """(Alice) send detected rounds and basis settings to Bob."""
        detected = sorted(self.owner.records.keys())
        settings = [self.owner.records[i][0] for i in detected]
        msg = EntQKDMessage(EntQKDMsgType.BASIS_ANNOUNCE, self.peer_proto,
                            detected=detected, settings=settings)
        self.owner.send_message(self.peer_node, msg)

    # ----- message dispatch -------------------------------------------------
    def received_message(self, src: str, msg: EntQKDMessage):
        if msg.msg_type is EntQKDMsgType.BASIS_ANNOUNCE:
            # common preamble of the sifting step, shared by BBM92 and E91:
            # index Alice's settings and select rounds detected on BOTH sides
            alice_detected = msg.kwargs["detected"]
            alice_settings = dict(zip(alice_detected, msg.kwargs["settings"]))
            common = [i for i in alice_detected if i in self.owner.records]
            self._bob_sift(common, alice_settings)
        elif msg.msg_type is EntQKDMsgType.SIFT_ANNOUNCE:
            # common closing step: adopt Bob's key rounds, then let the
            # subclass compute its security metric (QBER or CHSH)
            key_rounds = msg.kwargs["key_rounds"]
            self.key_rounds = key_rounds
            self.key = [self._bit(i) for i in key_rounds]
            self._alice_finish(msg)
        return True   # contract of Node.receive_message

    # ----- protocol-specific hooks ------------------------------------------
    def _bob_sift(self, common: list, alice_settings: dict):
        """(Bob) sift ``common`` rounds and reply with SIFT_ANNOUNCE.

        Args:
            common: rounds detected by BOTH parties.
            alice_settings: round -> Alice's basis setting index.
        """
        raise NotImplementedError

    def _alice_finish(self, msg: EntQKDMessage):
        """(Alice) compute protocol metrics; the key was already adopted."""
        raise NotImplementedError


def pair_entanglement_protocols(sender: BaseEntanglementQKD,
                                receiver: BaseEntanglementQKD,
                                anti_correlated: bool = None) -> None:
    """Pair two entanglement-QKD protocol instances (BB84-style helper).

    Args:
        sender: protocol instance acting as Alice (starts the classical phase).
        receiver: protocol instance acting as Bob (performs the sifting).
        anti_correlated: if given, overrides both instances (True for the
            |Psi-> source; False for |Phi+>).
    """
    assert type(sender) is type(receiver), \
        "cannot pair different entanglement-QKD protocols"
    sender.role = 0
    receiver.role = 1
    sender.peer_node = receiver.owner.name
    sender.peer_proto = receiver.name
    receiver.peer_node = sender.owner.name
    receiver.peer_proto = sender.name
    if anti_correlated is not None:
        sender.anti_correlated = anti_correlated
        receiver.anti_correlated = anti_correlated

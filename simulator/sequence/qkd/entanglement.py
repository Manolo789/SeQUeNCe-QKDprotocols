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

Shared infrastructure for entanglement-based QKD (BBM92 / E91).

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
import math
from math import cos, sin, sqrt
from typing import TYPE_CHECKING

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..protocol import StackProtocol
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
                 dst_alice: str, dst_bob: str, num_rounds: int = 0,
                 frequency: float = 1e6, wavelength_nm: float = 810,
                 bell_state: str = "psi_minus", mean_photon_num=None):
        super().__init__(name, timeline)
        self.owner_node = owner_node
        self.dst_alice = dst_alice
        self.dst_bob = dst_bob
        #: default train length (rounds); the actual value is normally passed
        #: to :meth:`start` by the protocol, which derives it from ``keysize``.
        self.num_rounds = num_rounds
        self.frequency = frequency
        self.wavelength_nm = wavelength_nm
        self.state = BELL_STATES[bell_state]
        self.mean_photon_num = mean_photon_num
        self.emitted = 0
        #: index of the train being emitted (stamped on every photon so a
        #: straggler from train k-1 is never recorded as a round of train k).
        self.train_index = 0

    def init(self):
        pass

    @property
    def period(self) -> int:
        """Emission period [ps] (one round per period)."""
        return int(round(1e12 / self.frequency))

    def start(self, start_time: int, num_rounds: int = None,
              train_index: int = None) -> int:
        """Schedule the emission of ONE train starting at ``start_time``.

        Args:
            start_time (int): simulation time [ps] of round 0 of the train.
            num_rounds (int): rounds in this train (default ``self.num_rounds``).
            train_index (int): tag stamped on the photons of this train.

        Returns:
            int: simulation time [ps] of the LAST emission of the train.
        """
        rounds = self.num_rounds if num_rounds is None else int(num_rounds)
        if train_index is not None:
            self.train_index = int(train_index)
        period = self.period
        for i in range(rounds):
            process = Process(self, "emit_round", [i])
            self.timeline.schedule(Event(start_time + i * period, process))
        return start_time + max(rounds - 1, 0) * period

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
        p_alice.train_index = self.train_index
        p_bob.train_index = self.train_index
        self.owner_node.send_qubit(self.dst_alice, p_alice)
        self.owner_node.send_qubit(self.dst_bob, p_bob)
        self.emitted += 1


# ===========================================================================
# Application: base class for entanglement-based QKD protocols
# ===========================================================================

class BaseEntanglementQKD(StackProtocol):
    """Classical layer common to BBM92 and E91 (keysize-oriented driver).

    Alice announces her bases and detected rounds (BASIS_ANNOUNCE). Bob
    sifts and replies (SIFT_ANNOUNCE); the subclass defines what Bob
    computes (:meth:`_bob_sift`) and how Alice finishes
    (:meth:`_alice_finish`).

    Mirrors the conventions of BB84/B92/COW in this repository:
      * ``role`` is -1 until :func:`~sequence.qkd.BBM92.pair_bbm92_protocols`
        (or the E91 analog) is called: 0 = Alice (initiator), 1 = Bob;
        ``another`` points at the peer protocol instance (BB84 convention).
      * The measuring hardware lives in the owner node
        (``topology.node.MeasuringNode``), just like the light source and
        detectors live in ``QKDNode`` for the prepare-and-measure protocols.
      * Key generation is requested with :meth:`push` ``(keysize, key_num,
        run_time)`` -- the SAME entry point as ``BB84.push`` -- and the
        SAME metric attributes are exported (``error_rates``,
        ``throughputs``, ``latency``, ``sifted_bits_length``,
        ``send_bits_length``), so both families are post-processed by the
        one shared estimator in ``QKD_Extension._collect_metrics``.

    Keysize-oriented operation (replaces the old fixed "number of rounds"):
      1. one TRAIN = ``rounds_per_train`` emission rounds of the Bell-pair
         source, derived from the requested key size exactly as BB84 derives
         its pulse train from ``light_time`` (``keysize / mean_photon_num``);
      2. after each train the classical phase runs (basis announcement +
         sifting) and the sifted bits are APPENDED to ``key_bits``;
      3. whenever ``len(key_bits) >= keysize`` a key is extracted (and the
         per-key metrics are recorded), exactly as in ``BB84``;
      4. trains repeat until ``run_time`` / the timeline stop time expires
         or ``key_num`` keys have been produced.

    The total number of emission rounds is therefore a DERIVED quantity
    (``num_rounds = num_trains * rounds_per_train``), not an input knob.

    If ``anti_correlated=True`` (state |Psi->) Bob flips his key/sample
    bits so that both keys coincide (Kržič Eq. 3.1).

    Class attributes (overridden by subclasses):
        ANGLES_ALICE (list[float]): analyser angles for Alice [degrees].
        ANGLES_BOB (list[float]): analyser angles for Bob [degrees].

    Attributes:
        owner (MeasuringNode): node the protocol is attached to.
        role (int): -1 unpaired, 0 Alice, 1 Bob.
        another (BaseEntanglementQKD): peer protocol instance.
        peer_node (str): name of the peer node.
        peer_proto (str): name of the peer protocol instance.
        anti_correlated (bool): True for the |Psi-> source.
        source (BellPairSource): the (untrusted) source driven by Alice's
            instance; a simulation-side handle, set by :meth:`attach_source`.
        train_key_bits (list[int]): sifted bits of the CURRENT train.
        key_bits (list[int]): sifted bits accumulated across trains.
        key (int): last extracted key (BB84 convention: bits packed as int).
        key_rounds (list[int]): round indices contributing to the last train.
        keysize (int): requested key length [bits].
        keys_left (int): keys still to be generated.
        rounds_per_train (int): emission rounds per train.
        num_rounds (int): DERIVED total of emission rounds requested so far.
        num_trains (int): number of trains launched so far.
        send_bits_length (int): denominator of R_s / R_sk -- qubits (pairs)
            emitted per train. Same role as in BB84.
        sifted_bits_length (list[int]): sifted bits of the train that closed
            each key (same convention as BB84).
        error_rates (list[float]): QBER of each extracted key.
        throughputs (list[float]): sifted-key throughput [bits/s] per key.
        latency (float): time to the first key [s].
        metrics (dict): protocol metrics of the last train
            (qber / chsh_S / sifted_len ...).
        chsh_values (list[float]): E91 CHSH parameter per train.
    """

    #: analyser angles in degrees; subclasses must override.
    ANGLES_ALICE: list = []
    ANGLES_BOB: list = []

    #: extra margin [ps] required to fit a whole train + classical phase
    #: inside the remaining simulation time before launching it.
    TRAIN_GUARD_PS: int = 10 ** 6

    def __init__(self, owner: "MeasuringNode", name: str,
                 peer_node_name: str = None, peer_protocol_name: str = None,
                 anti_correlated: bool = True):
        super().__init__(owner, name)
        self.role = -1                          # set by the pair_* function
        self.another: "BaseEntanglementQKD" = None
        self.peer_node = peer_node_name
        self.peer_proto = peer_protocol_name
        self.anti_correlated = anti_correlated

        # simulation-side handle to the untrusted source (Alice's instance)
        self.source: "BellPairSource" = None

        # ---- per-train state -------------------------------------------
        self.train_index: int = 0
        self.train_key_bits: list = []
        self.key_rounds: list = []
        self.metrics: dict = {}

        # ---- keysize-oriented accumulation (BB84 contract) -------------
        self.key_bits: list = []
        self.key: int = None
        self.keysize: int = 0
        self.keys_left: int = 0
        self.rounds_per_train: int = 0
        self.end_run_time: float = math.inf
        self.working: bool = False
        self.ready: bool = True

        # ---- metrics consumed by the SHARED estimator ------------------
        self.latency: float = 0.0
        self.throughputs: list = []
        self.error_rates: list = []
        self.sifted_bits_length: list = []
        self.send_bits_length: int = 0
        self.last_key_time: int = 0

        # ---- derived diagnostics ---------------------------------------
        self.num_rounds: int = 0
        self.num_trains: int = 0
        self.chsh_values: list = []

    def init(self):
        pass

    def pop(self, **kwargs):
        """Downward stack interface (unused: sifting is the bottom layer)."""
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

    # ==================================================================
    #  Key-generation driver (keysize-oriented; mirrors BB84.push)
    # ==================================================================
    def attach_source(self, source: "BellPairSource") -> None:
        """Give the protocol the handle used to trigger the pair trains.

        Charlie is an UNTRUSTED relay: the handle is a simulation-side
        convenience (it lets the trains be sized from ``keysize`` and keeps
        the per-train memory bounded), not a trust assumption -- Alice only
        ever uses her own detections and the public classical messages.
        """
        self.source = source
        if self.another is not None:
            self.another.source = source

    def push(self, length: int, key_num: int = 1, run_time=math.inf,
             rounds_per_train: int = None) -> None:
        """Request key generation (same entry point as ``BB84.push``).

        Args:
            length (int): length of the key(s) to generate [bits].
            key_num (int): number of keys to generate.
            run_time (int): max simulation time [ps] allowed for the request.
            rounds_per_train (int): emission rounds per train. Default:
                ``round(length / mean_photon_num)``, the entanglement analog
                of the BB84 pulse train (``light_time * frequency``).
        """
        if self.role != 0:
            raise AssertionError("generate key must be called from Alice")
        if self.source is None:
            raise AssertionError(
                "attach_source() must be called before push() "
                "(the pair source drives the emission trains)")
        if self.another is None:
            raise AssertionError("protocols were not paired")

        now = self.owner.timeline.now()
        if rounds_per_train is None:
            mu = self.source.mean_photon_num or 1.0
            rounds_per_train = max(1, int(round(length / mu)))

        for p in (self, self.another):
            p.keysize = int(length)
            p.rounds_per_train = int(rounds_per_train)
            p.end_run_time = run_time + now
        self.keys_left = int(key_num) if key_num != math.inf else math.inf

        if self.ready:
            self.ready = False
            self.start_protocol()

    def start_protocol(self) -> None:
        """Reset both sides and launch the first train."""
        for p in (self, self.another):
            p.key_bits = []
            p.train_key_bits = []
            p.key = None
            p.latency = 0.0
            p.working = True
            # denominator of R_s / R_sk: qubits (pairs) emitted per train
            p.send_bits_length = self.rounds_per_train
        self.last_key_time = self.owner.timeline.now()
        self.begin_train()

    # ----- one train --------------------------------------------------
    def begin_train(self) -> None:
        """Emit one train of pairs and schedule its classical phase."""
        tl = self.owner.timeline
        now = tl.now()
        if (not self.working) or self.keys_left < 1 or now >= self.end_run_time:
            self._stop()
            return

        rounds = self.rounds_per_train
        span = max(rounds - 1, 0) * self.source.period
        q_delay = self._quantum_delay()
        c_delay = self._classical_delay()

        # Only launch the train if the WHOLE cycle (quantum window + the
        # two classical legs) still fits: a truncated train would bias the
        # metrics (rounds counted in the denominator whose sifted bits
        # never arrive).
        cycle = span + q_delay + 2 * c_delay + self.TRAIN_GUARD_PS
        if now + cycle >= min(tl.stop_time, self.end_run_time):
            self._stop()
            return

        self.train_index += 1
        self.another.train_index = self.train_index

        # fresh analyser settings + empty records on both measuring nodes
        self.owner.start_train(rounds, self.train_index)
        self.another.owner.start_train(rounds, self.train_index)

        self.source.start(now, num_rounds=rounds,
                          train_index=self.train_index)

        # DERIVED quantity: rounds are counted, never configured
        self.num_rounds += rounds
        self.another.num_rounds = self.num_rounds
        self.num_trains += 1
        self.another.num_trains = self.num_trains

        t_quantum_end = now + span + q_delay + 1
        tl.schedule(Event(t_quantum_end,
                          Process(self.owner, "apply_noise_counts", [])))
        tl.schedule(Event(t_quantum_end + 1,
                          Process(self.another.owner, "apply_noise_counts", [])))
        tl.schedule(Event(t_quantum_end + 2,
                          Process(self, "announce_bases", [])))

    def _quantum_delay(self) -> int:
        """Worst-case Charlie -> receiver propagation delay [ps]."""
        node = self.source.owner_node
        delay = 0
        for channel in node.qchannels.values():
            d = channel.delay
            seg2 = getattr(channel, "_seg2", None)   # EveQuantumChannel
            if seg2 is not None:
                d += seg2.delay
            delay = max(delay, d)
        return int(delay)

    def _classical_delay(self) -> int:
        """Worst-case one-way delay [ps] of the Alice <-> Bob classical link."""
        delays = [c.delay for c in self.owner.cchannels.values()]
        delays += [c.delay for c in self.another.owner.cchannels.values()]
        return int(max(delays)) if delays else 0

    def _stop(self) -> None:
        """Terminate the key-generation request on both sides."""
        self.working = False
        if self.another is not None:
            self.another.working = False
        # Without a post-processing stack there is nothing left to deliver:
        # stop the timeline so noise chains do not keep replaying until the
        # full horizon. With a stack, the authentication layer stops the
        # timeline after the last key is delivered.
        if self.role == 0 and not self.upper_protocols:
            self.owner.timeline.stop()

    # ----- end of a train: accumulate and extract keys ------------------
    def _train_complete(self) -> None:
        """(Alice) fold the train into ``key_bits`` and extract full keys.

        Faithful to ``BB84.received_message``: the sifted bits accumulate,
        a key is emitted whenever the accumulator reaches ``keysize``, and
        the per-key metrics use the SAME definitions (QBER over the
        extracted key, throughput and latency measured from the previous
        key, ``sifted_bits_length`` of the train that closed the key).
        """
        tl = self.owner.timeline
        now = tl.now()

        # Alice's and Bob's bits are indexed by the same round list, so a
        # truncation to the shorter one keeps them aligned.
        n_sifted = min(len(self.train_key_bits),
                       len(self.another.train_key_bits))
        self.key_bits.extend(self.train_key_bits[:n_sifted])
        self.another.key_bits.extend(self.another.train_key_bits[:n_sifted])

        keysize = self.keysize
        if keysize > 0 and len(self.key_bits) >= keysize and self.keys_left > 0:
            elapsed = now - self.last_key_time
            throughput = (keysize * 1e12 / elapsed) if elapsed > 0 else 0.0

            while len(self.key_bits) >= keysize and self.keys_left > 0:
                self.sifted_bits_length.append(n_sifted)
                self.set_key()
                self.another.set_key()
                # deliver the sifted key to the classical post-processing
                # stack (parameter estimation -> EC -> EE -> PA -> auth),
                # SAME convention as BB84/B92/COW
                self._pop(info=self.key, length=keysize)
                self.another._pop(info=self.another.key, length=keysize)

                if self.latency == 0:
                    self.latency = elapsed * 1e-12
                self.throughputs.append(throughput)
                self.error_rates.append(self._key_error_rate())

                self.keys_left -= 1

            self.last_key_time = now

        if self.keys_left < 1:
            self._stop()
            return
        tl.schedule(Event(now + 1, Process(self, "begin_train", [])))

    def set_key(self) -> None:
        """Pack the first ``keysize`` accumulated bits into ``self.key``."""
        bits = self.key_bits[0:self.keysize]
        del self.key_bits[0:self.keysize]
        self.key = int("".join(str(x) for x in bits), 2)

    def _key_error_rate(self) -> float:
        """QBER of the key just extracted (identical to the BB84 recipe)."""
        if not self.keysize:
            return 0.0
        key_diff = self.key ^ self.another.key
        num_errors = 0
        while key_diff:
            key_diff &= key_diff - 1
            num_errors += 1
        return num_errors / self.keysize

    # ==================================================================
    #  Classical phase
    # ==================================================================
    def announce_bases(self):
        """(Alice) send detected rounds and basis settings to Bob."""
        detected = sorted(self.owner.records.keys())
        settings = [self.owner.records[i][0] for i in detected]
        msg = EntQKDMessage(EntQKDMsgType.BASIS_ANNOUNCE, self.peer_proto,
                            detected=detected, settings=settings,
                            train_index=self.train_index)
        self.owner.send_message(self.peer_node, msg)

    # ----- message dispatch -------------------------------------------------
    def received_message(self, src: str, msg: EntQKDMessage):
        if msg.msg_type is EntQKDMsgType.BASIS_ANNOUNCE:
            # common preamble of the sifting step, shared by BBM92 and E91:
            # index Alice's settings and select rounds detected on BOTH sides
            alice_detected = msg.kwargs["detected"]
            alice_settings = dict(zip(alice_detected, msg.kwargs["settings"]))
            common = [i for i in alice_detected if i in self.owner.records]
            self.train_key_bits = []
            self._bob_sift(common, alice_settings)
        elif msg.msg_type is EntQKDMsgType.SIFT_ANNOUNCE:
            # common closing step: adopt Bob's key rounds, then let the
            # subclass compute its security metric (QBER or CHSH)
            key_rounds = msg.kwargs["key_rounds"]
            self.key_rounds = key_rounds
            self.train_key_bits = [self._bit(i) for i in key_rounds]
            self._alice_finish(msg)
            # keysize-oriented accumulation (Alice drives the trains)
            self._train_complete()
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
    # BB84 convention: each instance keeps a handle to its peer, used by the
    # keysize-oriented driver (key extraction / QBER of the extracted key).
    sender.another = receiver
    receiver.another = sender
    sender.peer_node = receiver.owner.name
    sender.peer_proto = receiver.name
    receiver.peer_node = sender.owner.name
    receiver.peer_proto = sender.name
    if anti_correlated is not None:
        sender.anti_correlated = anti_correlated
        receiver.anti_correlated = anti_correlated

"""
==================================================================
Extension of the COW protocol to the SeQUeNCe simulator -- License
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

"""Definition of COW protocol implementation.

This code is inspired by the original code for the BB84 protocol in https://github.com/sequence-toolbox/SeQUeNCe/blob/master/sequence/qkd/BB84.py
"""

import math
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Deque, Tuple

if TYPE_CHECKING:
    from ..topology.node import QKDNode

import numpy

from ..message import Message
from ..protocol import StackProtocol
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log
from ..utils.encoding_cow import build_cow_state_list, time_bin_cow


def pair_cow_protocols(sender: "COW", receiver: "COW") -> None:
    """Function to pair COW protocol instances.

    Args:
        sender (COW): protocol instance sending qubits (Alice).
        receiver (COW): protocol instance receiving qubits (Bob).
    """

    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1


class COWMsgType(Enum):
    """Defines possible message types for COW."""

    BEGIN_PHOTON_PULSE = auto()  # Alice → Bob  : parameters of the upcoming pulse
    RECEIVED_QUBITS = auto()     # Bob   → Alice : indices Bob detected + raw bits
    DECOY_POSITIONS   = auto()   # Alice → Bob  : which indices were decoy sequences
    SIFTED_INDICES    = auto()   # Bob   → Alice : which non-decoy indices to keep


class COWMessage(Message):
    """Classical message exchanged between Alice and Bob during COW.

    Attributes:
        msg_type (COWMsgType): message type tag.
        receiver (str): name of the destination protocol instance.
        frequency (float): light-source clock frequency in Hz (BEGIN_PHOTON_PULSE only).
        light_time (float): burst duration in s (BEGIN_PHOTON_PULSE only).
        start_time (int): burst start time in ps (BEGIN_PHOTON_PULSE only).
        wavelength (float): photon wavelength in nm (BEGIN_PHOTON_PULSE only).
        decoy_rate (float): fraction of symbols used as decoys (BEGIN_PHOTON_PULSE only).
        indices (list[int]): detected / sifted symbol indices (RECEIVED_QUBITS, SIFTED_INDICES).
        bits (list[int]): raw detected bits (RECEIVED_QUBITS).
        decoy_indices (list[int]): positions of decoy symbols (DECOY_POSITIONS).
        """

    def __init__(self, msg_type: COWMsgType, receiver: str, **kwargs):
        Message.__init__(self, msg_type, receiver)
        self.protocol_type = COW
        if self.msg_type is COWMsgType.BEGIN_PHOTON_PULSE:
            self.frequency = kwargs["frequency"]
            self.light_time = kwargs["light_time"]
            self.start_time = kwargs["start_time"]
            self.wavelength = kwargs["wavelength"]
            self.decoy_rate = kwargs.get("decoy_rate", 0.1)
        elif self.msg_type is COWMsgType.RECEIVED_QUBITS:
            self.indices = kwargs["indices"]
            self.bits = kwargs["bits"]
        elif self.msg_type is COWMsgType.DECOY_POSITIONS:
            self.decoy_indices = kwargs["decoy_indices"]
        elif self.msg_type is COWMsgType.SIFTED_INDICES:
            self.indices = kwargs["indices"]
        else:
            raise Exception(f"COW generated invalid message type {msg_type}")


class COW(StackProtocol):
    """Coherent One-Way (COW) QKD protocol.

    Attributes:
        owner (QKDNode): attached node.
        name (str): protocol label.
        role (int): 0 = Alice, 1 = Bob.
        working (bool): True during an active key-generation run.
        ready (bool): True when Alice is idle.
        light_time (float): current burst duration in s.
        ls_freq (float): light-source frequency in Hz.
        start_time (int): burst start time in ps.
        decoy_rate (float): fraction of symbols that are decoys.
        bit_lists (list[list[int]]): Alice's emitted bits per burst.
        decoy_positions (list[list[int]]): Alice's decoy positions per burst.
        _pending_bob_indices (list[int]): Bob's detections buffered at Alice.
        _pending_bob_bits (list[int]): Bob's raw bits buffered at Alice.
        key (int): most recent key as integer.
        key_bits (list[int]): accumulated sifted key bits.
        another (COW): partner protocol.
        key_lengths (list[int]): requested key-length queue.
        keys_left_list (list[int]): remaining keys per request.
        end_run_times (list[int]): deadlines per request.
        latency (float): time to first key in s.
        last_key_time (int): timeline time of last key event in ps.
        sifted_bits_length (list[int]): sifted-key lengths per run.
        throughputs (list[float]): key throughputs in bits/s.
        error_rates (list[float]): QBERs per run.
        visibility (list[float]): monitoring visibility per burst
            (Michelson-based when QSDetectorCOW is used).
    """
    # Minimum acceptable visibility — below this threshold the protocol
    # should abort (not enforced here; left to higher-layer logic).
    VISIBILITY_THRESHOLD = 0.9

    def __init__(self, owner: "QKDNode", name: str, lightsource: str, qsdetector: str, role=-1):
        """Constructor for COW class.

        Args:
            owner (QKDNode): node hosting protocol instance.
            name (str): name of protocol instance.
            lightsource (str): name of lightsource for QKD
            qsdetector (str): name of QSDetector for QKD

        Keyword Args:
            role (int): 0/1 role defining Alice and Bob protocols (default -1).
        """

        if owner is None:  # used only for unit test purposes
            return
        super().__init__(owner, name)
        self.ls_name = lightsource
        self.qsd_name = qsdetector
        self.role = role

        # State flags
        self.working = False
        self.ready = True  # (for Alice) not currently processing a generate_key request

        # Timing / hardware parameters
        self.light_time = 0  # time to use laser (measured in s)
        self.ls_freq = 0  # frequency of light source
        self.start_time = 0  # start time of light pulse
        self.photon_delay = 0  # time delay of photon (including dispersion) (ps)
        self.decoy_rate = 0.1    # fraction of symbols used as decoys


        # Alice-side buffers (populated in begin_photon_pulse)
        self.bit_lists = None
        self.decoy_positions = None
        
        self._pending_bob_indices: List[int] = []
        self._pending_bob_bits:    List[int] = []

        self._burst_queue: Deque[Tuple[List[int], List[int]]] = deque()

        # Key storage
        self.key = 0  # key as int
        self.key_bits = None  # key as list of bits
        self.another = None

        # Request queues (mirrors BB84)
        self.key_lengths = []  # desired key lengths (from parent)
        self.keys_left_list = []
        self.end_run_times = []

        # metrics
        self.latency = 0  # measured in seconds
        self.last_key_time = 0
        self.sifted_bits_length = []
        self.send_bits_length = 0
        self.throughputs = [] # measured in bits/sec
        self.error_rates = []
        self.visibility = [] # monitoring-line visibility
        
        # Cache da basis_list para set_measure_basis_list
        self._cached_basis_list: List[int] = []
        self._cached_basis_n: int = 0

    def pop(self, detector_index: int, time: int) -> None:
        """Method to receive detection events (currently unused)."""

        assert 0

    def push(self, length: int, key_num: int, run_time=math.inf) -> None:
        """Method to receive requests for key generation.

        Args:
            length (int): length of key to generate.
            key_num (int): number of keys to generate.
            run_time (int): max simulation time allowed for key generation (default inf).

        Side Effects:
            Will potentially invoke `start_protocol` method to start operations.
        """

        if self.role != 0:
            raise AssertionError("generate key must be called from Alice")

        log.logger.info(self.name + f" generating keys [COW], keylen={length}, keynum={key_num}")

        self.key_lengths.append(length)
        self.another.key_lengths.append(length)
        self.keys_left_list.append(key_num)
        end_run_time = run_time + self.owner.timeline.now()
        self.end_run_times.append(end_run_time)
        self.another.end_run_times.append(end_run_time)

        if self.ready:
            self.ready = False
            self.working = True
            self.another.working = True
            self.start_protocol()

    def start_protocol(self) -> None:
        """Method to start protocol.

        When called, this method will begin the process of key generation.
        Parameters for hardware will be calculated, and a `begin_photon_pulse` method scheduled.

        Side Effects:
            Will schedule future `begin_photon_pulse` event.
            Will send a BEGIN_PHOTON_PULSE method to other protocol instance.
        """

        log.logger.debug(self.name + " starting protocol")

        if len(self.key_lengths) > 0:
            # reset buffers for self and another

            self.bit_lists = []
            self.decoy_positions = []
            self.key_bits = []
            self._pending_bob_indices  = []
            self._pending_bob_bits     = []

            self.another.bit_lists = []
            self.another.decoy_positions = []
            self.another.key_bits = []
            self.another._burst_queue = deque()

            self.latency = 0
            self.another.latency = 0
            self.working = True
            self.another.working = True

            ls = self.owner.components[self.ls_name]
            self.ls_freq = ls.frequency

            # calculate light time based on key length
            # COW uses 2 time slots per symbol, so the burst covers twice as many
            # raw time-bin periods as the number of requested key bits.
            self.light_time = (self.key_lengths[0] / (self.ls_freq * ls.mean_photon_num)) * 2

            # Invalida cache da basis_list ao mudar light_time
            self._cached_basis_n = 0

            # send message that photon pulse is beginning, then send bits
            self.start_time = int(self.owner.timeline.now()) + round(self.owner.cchannels[self.another.owner.name].delay)
            message = COWMessage(COWMsgType.BEGIN_PHOTON_PULSE, self.another.name,
                                  frequency=self.ls_freq, light_time=self.light_time,
                                  start_time=self.start_time, wavelength=ls.wavelength,
                                  decoy_rate=self.decoy_rate)
            self.owner.send_message(self.another.owner.name, message)

            process = Process(self, "begin_photon_pulse", [])
            event = Event(self.start_time, process)
            self.owner.timeline.schedule(event)

            self.last_key_time = self.owner.timeline.now()
        else:
            self.ready = True

    def begin_photon_pulse(self) -> None:
        """Method to begin sending photons.

        Will calculate qubit parameters and invoke lightsource emit method.
        Also records bits sent for future processing.

        Side Effects:
            Will set destination of photons for local node.
            Will invoke emit method of node lightsource.
            Will schedule another `begin_photon_pulse` event after the emit period.
        """
        
        log.logger.debug(self.name + " starting photon pulse")
        
        if self.working and self.owner.timeline.now() < self.end_run_times[0]:
            self.owner.destination = self.another.owner.name

            # control hardware
            lightsource = self.owner.components[self.ls_name]

            # Number of COW symbols in this burst
            num_symbols = int(self.light_time * self.ls_freq / 2)

            # Randomly assign data bits and flag decoy symbols
            bit_list  = numpy.random.choice([0, 1], num_symbols)
            is_decoy  = numpy.random.random(num_symbols) < self.decoy_rate
            
            decoy_pos = [int(i) for i, d in enumerate(is_decoy) if d]
            self.bit_lists.append(bit_list.tolist())
            self.decoy_positions.append(decoy_pos)
            self.send_bits_length = num_symbols
            
            state_list = build_cow_state_list(bit_list.tolist(), is_decoy.tolist())
            lightsource.emit(state_list)

            # schedule another
            self.start_time = self.owner.timeline.now()
            process = Process(self, "begin_photon_pulse", [])
            event = Event(self.start_time + int(round(self.light_time * 1e12)), process)
            self.owner.timeline.schedule(event)

        else:
            self.working = False
            self.another.working = False

            self.key_lengths.pop(0)
            self.keys_left_list.pop(0)
            self.end_run_times.pop(0)
            self.another.key_lengths.pop(0)
            self.another.end_run_times.pop(0)

            # wait for quantum channel to clear of photons, then start protocol
            time = self.owner.timeline.now() + self.owner.qchannels[self.another.owner.name].delay + 1
            process = Process(self, "start_protocol", [])
            event = Event(time, process)
            self.owner.timeline.schedule(event)
            

    def set_measure_basis_list(self) -> None:
        """Configure Bob's detector for passive direct detection (COW).

        COW requires no active basis switching.  All slots are detected
        directly by the DB detector.  For :class:`QSDetectorCOW` this call
        is a no-op; for backward-compatible ``QSDetectorTimeBin`` usage a
        constant all-zero basis list is set.
        """
        log.logger.debug(self.name + " setting COW measurement (beamsplitter model)")
        
        num_slots = int(self.light_time * self.ls_freq)
        if num_slots != self._cached_basis_n:
            self._cached_basis_list = [0] * num_slots
            self._cached_basis_n    = num_slots
        self.owner.components[self.qsd_name].set_basis_list(self._cached_basis_list, self.start_time, self.ls_freq)


    def end_photon_pulse(self) -> None:
        """Method to process sent qubits."""

        log.logger.debug(self.name + " ending photon pulse")

        if self.working and self.owner.timeline.now() < self.end_run_times[0]:

            # get_bits returns one entry per raw time-bin slot using time_bin encoding.
            # -1 → no detection or ambiguous; 0 → early-bin detection; 1 → late-bin detection
            raw_bits = self.owner.get_bits(self.light_time, self.start_time, self.ls_freq, self.qsd_name)

            detected_indices: List[int] = []
            detected_bits:    List[int] = []

            # Otimizado — operações numpy vetorizadas
            raw = numpy.array(raw_bits)
            early = raw[0::2]   # slots pares
            late  = raw[1::2]   # slots ímpares

            mask_bit1 = (early == -1) & (late != -1)
            mask_bit0 = (early != -1) & (late == -1)

            indices_bit1 = numpy.where(mask_bit1)[0]
            indices_bit0 = numpy.where(mask_bit0)[0]

            all_indices = numpy.concatenate([indices_bit0, indices_bit1])
            all_bits    = numpy.concatenate([numpy.zeros(len(indices_bit0), dtype=int),
                               numpy.ones(len(indices_bit1), dtype=int)])

            order = numpy.argsort(all_indices)
            detected_indices = all_indices[order].tolist()
            detected_bits    = all_bits[order].tolist()
            
            # ── CORREÇÃO: enfileira dados deste burst (não acumula na lista) ──
            self._burst_queue.append((detected_indices, detected_bits))

            # ---- Read Michelson visibility if QSDetectorCOW is available ----
            qsd = self.owner.components[self.qsd_name]
            if hasattr(qsd, "get_monitoring_visibility"):
                v = qsd.get_monitoring_visibility()
                self.visibility.append(v)
                print(f"Visibility = {v}")
                log.logger.info(self.name + f" [COW] Michelson visibility = {v:.4f} "+f"(threshold = {self.VISIBILITY_THRESHOLD})")
                if v < self.VISIBILITY_THRESHOLD:
                    log.logger.warning(self.name + " [COW] visibility below threshold — "+"possible eavesdropping or interferometer drift!")

            self.start_time = self.owner.timeline.now()

            # schedule another if necessary
            if self.owner.timeline.now() + self.light_time * 1e12 - 1 < self.end_run_times[0]:
                # schedule another
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + int(round(self.light_time * 1e12) - 1), process)
                self.owner.timeline.schedule(event)
            
            
            # send message that we got photons
            message = COWMessage(COWMsgType.RECEIVED_QUBITS, self.another.name, indices=detected_indices, bits=detected_bits)
            self.owner.send_message(self.another.owner.name, message)
    
    def received_message(self, src: str, msg: "Message") -> None:
        """Method to receive messages.

        Will perform different processing actions based on the message received.

        Args:
            src (str): source node sending message.
            msg (Message): message received.
        """

        if self.working and self.owner.timeline.now() < self.end_run_times[0]:
            if msg.msg_type is COWMsgType.BEGIN_PHOTON_PULSE:  # (current node is Bob): start to receive photons
                self.ls_freq = msg.frequency
                self.light_time = msg.light_time
                self.decoy_rate = msg.decoy_rate

                log.logger.debug(self.name + f" received BEGIN_PHOTON_PULSE, ls_freq={self.ls_freq}, light_time={self.light_time}")

                self.start_time = int(msg.start_time) + self.owner.qchannels[src].delay

                self._burst_queue     = deque()   # limpa fila ao iniciar nova rodada
                self._cached_basis_n  = 0

                self.set_measure_basis_list()

                # schedule end_photon_pulse()
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + round(self.light_time * 1e12) - 1, process)
                self.owner.timeline.schedule(event)

            elif msg.msg_type is COWMsgType.RECEIVED_QUBITS:  # (Current node is Alice): can send decoys positions
                log.logger.debug(self.name + " received RECEIVED_QUBITS message")
                self._pending_bob_indices = list(msg.indices)
                self._pending_bob_bits    = list(msg.bits)

                # Reveal which symbols were decoys
                decoy_pos = self.decoy_positions.pop(0) if self.decoy_positions else []
                message = COWMessage(COWMsgType.DECOY_POSITIONS, self.another.name, decoy_indices=decoy_pos)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.DECOY_POSITIONS:  # (Current node is Bob): compare bases
                log.logger.debug(self.name + " received DECOY_POSITIONS")

                if not self._burst_queue:
                    return

                burst_indices, burst_bits = self._burst_queue.popleft()

                decoy_set      = set(msg.decoy_indices)
                sifted_indices: List[int] = []
                app_si = sifted_indices.append
                app_kb = self.key_bits.append

                for i, idx in enumerate(burst_indices):
                    if idx not in decoy_set:
                        app_si(idx)
                        app_kb(burst_bits[i])

                # Send sifted (non-decoy) indices to Alice for key alignment
                message = COWMessage(COWMsgType.SIFTED_INDICES, self.another.name, indices=sifted_indices)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.SIFTED_INDICES:  # (Current node is Alice): create key from matching indices
                log.logger.debug(self.name + " received SIFTED_INDICES")
                # parse matching indices
                sifted_set = set(msg.indices)

                # Retrieve the bit list emitted in the most recent burst
                bit_list = self.bit_lists.pop(0) if self.bit_lists else []

                for idx in msg.indices:
                    if idx < len(bit_list):
                        self.key_bits.append(bit_list[idx])

                # check if key long enough. If it is, truncate if necessary and call cascade
                if len(self.key_bits) >= self.key_lengths[0]:
                    
                    throughput = self.key_lengths[0] * 1e12 / max(self.owner.timeline.now() - self.last_key_time, 1)
                    while len(self.key_bits) >= self.key_lengths[0] and self.keys_left_list[0] > 0:
                        log.logger.info(self.name + " generated a valid key")
                        self.sifted_bits_length.append(len(msg.indices))
                        self.set_key()  # convert from binary list to int
                        self._pop(info=self.key)
                        self.another.set_key()
                        self.another._pop(info=self.another.key)

                        # for metrics
                        if self.latency == 0:
                            self.latency = (self.owner.timeline.now() - self.last_key_time) * 1e-12

                        self.throughputs.append(throughput)
                        
                        # Compute QBER by XOR-ing Alice's and Bob's keys
                        key_diff = self.key ^ self.another.key
                        num_errors = 0
                        while key_diff:
                            key_diff &= key_diff - 1
                            num_errors += 1
                        self.error_rates.append(num_errors / self.key_lengths[0])

                        self.keys_left_list[0] -= 1

                    self.last_key_time = self.owner.timeline.now()

                # check if we're done
                if self.keys_left_list[0] < 1:
                    self.working = False
                    self.another.working = False

    def set_key(self):
        """Method to convert `bit_list` field (list[int]) to a single key (int)."""

        key_bits = self.key_bits[0:self.key_lengths[0]]
        del self.key_bits[0:self.key_lengths[0]]
        self.key = int("".join(str(x) for x in key_bits), 2)  # convert from binary list to int

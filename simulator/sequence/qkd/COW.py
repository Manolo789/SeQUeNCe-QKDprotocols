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
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..topology.node import QKDNode

import numpy

from ..message import Message
from ..protocol import StackProtocol
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log


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

    BEGIN_PHOTON_PULSE = auto()   # Alice → Bob  : parameters of the upcoming pulse
    RECEIVED_QUBITS = auto()   # Bob   → Alice : indices Bob detected + raw bits
    #BASIS_LIST = auto()
    DECOY_POSITIONS   = auto()   # Alice → Bob  : which indices were decoy sequences
    #MATCHING_INDICES = auto()
    SIFTED_INDICES    = auto()   # Bob   → Alice : which non-decoy indices to keep


class COWMessage(Message):
    """Message used by COW protocols.

    This message contains all information passed between COW protocol instances.
    Messages of different types contain different information.

    Attributes:
        msg_type (COWMsgType): defines the message type.
        receiver (str): name of destination protocol instance.
        frequency (float): frequency for qubit generation (if `msg_type == BEGIN_PHOTON_PULSE`).
        light_time (float): lenght of time to send qubits (if `msg_type == BEGIN_PHOTON_PULSE`).
        start_time (int): simulation start time of qubit pulse (if `msg_type == BEGIN_PHOTON_PULSE`).
        wavelength (float): wavelength (in nm) of photons (if `msg_type == BEGIN_PHOTON_PULSE`).
        bases (list[int]): list of measurement bases (if `msg_type == BASIS_LIST`).
        indices (list[int]): list of indices for matching bases (if `msg_type == MATCHING_INDICES`).
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
            pass
        #elif self.msg_type is COWMsgType.BASIS_LIST:
        #    self.bases = kwargs["bases"]
        elif self.msg_type is COWMsgType.DECOY_POSITIONS:
            self.decoy_indices = kwargs["decoy_indices"]
        #elif self.msg_type is COWMsgType.MATCHING_INDICES:
        #    self.indices = kwargs["indices"]
        elif self.msg_type is COWMsgType.SIFTED_INDICES:
            self.indices = kwargs["indices"]
        else:
            raise Exception(f"COW generated invalid message type {msg_type}")


class COW(StackProtocol):
    """Implementation of COW protocol.

    The COW protocol uses photons to create a secure key between two QKD Nodes.

    Attributes:
        owner (QKDNode): node that protocol instance is attached to.
        name (str): label for protocol instance.
        role (int): determines if instance is "alice" or "bob" node.
        working (bool): shows if protocol is currently working on a key.
        ready (bool): used by alice to show if protocol currently processing a generate_key request.
        light_time (float): time to use laser (in s).
        start_time (int): simulation start time of key generation.
        photon_delay (int): time delay of photon (ps).
        
        cancelled basis_lists (list[int]): list of bases that qubits are sent in.
        
        bit_lists (list[int]): list of 0/1 qubits sent (in bases from basis_lists).
        key (int): generated key as an integer.
        key_bits (list[int]): generated key as a list of 0/1 bits.
        another (COW): other COW protocol instance (on opposite node).
        key_lengths (list[int]): list of desired key lengths.
        self.keys_left_list (list[int]): list of desired number of keys.
        self.end_run_times (list[int]): simulation time for end of each request.
    """

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
        self.decoy_rate   = 0.1    # fraction of symbols used as decoys
        #self.basis_lists = None

        # Alice-side buffers (populated in begin_photon_pulse)
        self.bit_lists = None
        self.decoy_positions = None   # list[list[int]]

        # Temporary buffers used during sifting (Alice side)
        self._pending_bob_indices: List[int] = []
        self._pending_bob_bits:    List[int] = []

        # Bob-side detection buffer (populated in end_photon_pulse)
        self.received_indices: List[int] = []
        self.received_bits:    List[int] = []

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
        self.throughputs = []  # measured in bits/sec
        self.error_rates = []

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

            self.bit_lists             = []
            self.decoy_positions       = []
            self.key_bits              = []
            self._pending_bob_indices  = []
            self._pending_bob_bits     = []

            self.another.bit_lists         = []
            self.another.decoy_positions   = []
            self.another.key_bits          = []
            self.another.received_indices  = []
            self.another.received_bits     = []
            
            self.latency         = 0
            self.another.latency = 0
            self.working         = True
            self.another.working = True

            ls = self.owner.components[self.ls_name]
            self.ls_freq = ls.frequency

            # calculate light time based on key length
            # COW uses 2 time slots per symbol, so the burst covers twice as many
            # raw time-bin periods as the number of requested key bits.
            self.light_time = (self.key_lengths[0] / (self.ls_freq * ls.mean_photon_num)) * 2

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
            encoding_type = lightsource.encoding_type

            # Number of COW symbols in this burst
            num_symbols = round(self.light_time * self.ls_freq / 2)

            # Randomly assign data bits and flag decoy symbols
            bit_list  = numpy.random.choice([0, 1], num_symbols)
            is_decoy  = numpy.random.random(num_symbols) < self.decoy_rate
            
            decoy_pos = [int(i) for i, d in enumerate(is_decoy) if d]
            self.bit_lists.append(bit_list.tolist())
            self.decoy_positions.append(decoy_pos)

            # Build the flat state list sent to the lightsource.
            # In time_bin encoding:
            #   bases[0][0] → |early⟩  (bit = 0, i.e. "early bin active")
            #   bases[0][1] → |late⟩   (bit = 1, i.e. "late  bin active")
            # We map:
            #   COW bit 0  → late  pulse  → (vacuum, bases[0][1])
            #   COW bit 1  → early pulse  → (bases[0][0], vacuum)
            #   COW decoy  → both pulses  → (bases[0][0], bases[0][1])
            # Vacuum is represented by the "0" state in time_bin:
            #   bases[0][0] encodes photon in early slot,
            #   so "no photon" at a slot means we skip emission for that slot.
            # Because LightSource.emit() sends one photon per entry (with
            # mean_photon_num governing actual emission probability), we use
            # the state that directs the photon to the correct slot,
            # and we insert None / bases[0][0] as a placeholder for vacuum.
            # The convention adopted here follows SeQUeNCe's time_bin helper:
            #   emit state (1, 0) → photon in early bin
            #   emit state (0, 1) → photon in late  bin
            early_state = encoding_type["bases"][0][0]  # |early⟩
            late_state  = encoding_type["bases"][0][1]  # |late⟩

            state_list = []
            for i in range(num_symbols):
                if is_decoy[i]:
                    # Decoy: coherent pulse in both bins
                    state_list.append(early_state)
                    state_list.append(late_state)
                elif bit_list[i] == 0:
                    # Bit 0: vacuum in early, pulse in late
                    state_list.append(early_state)   # "early" slot — vacuum encoded
                    state_list.append(late_state)    # "late"  slot — pulse
                else:
                    # Bit 1: pulse in early, vacuum in late
                    state_list.append(early_state)   # "early" slot — pulse
                    state_list.append(late_state)    # "late"  slot — vacuum encoded

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
        """Configure the detector for passive (basis-free) COW detection.

        In COW there is no basis reconciliation: Bob uses direct detection
        at all times (no measurement-basis switching).  We set a uniform
        all-zeros basis list so that the QSDetector's switch remains in
        the direct-detection position throughout the burst.
        """
        # Nome a função preservado para compatibilidade na documentação

        log.logger.debug(self.name + " setting COW measurement (direct detection, no basis)")

        num_slots = int(self.light_time * self.ls_freq)   # total time-bin slots
        basis_list = [0] * num_slots
        self.owner.set_basis_list(basis_list, self.start_time, self.ls_freq, self.qsd_name)

    def end_photon_pulse(self) -> None:
        """Method to process sent qubits."""

        log.logger.debug(self.name + " ending photon pulse")

        if self.working and self.owner.timeline.now() < self.end_run_times[0]:
            # get_bits returns one entry per raw time-bin slot using time_bin encoding.
            # -1 → no detection or ambiguous; 0 → early-bin detection; 1 → late-bin detection
            raw_bits  = self.owner.get_bits(self.light_time, self.start_time, self.ls_freq, self.qsd_name)
            num_symbols = len(raw_bits) // 2

            detected_indices = []
            detected_bits    = []
            decoy_candidates = []   # symbols where both bins fired (likely decoys)

            for i in range(num_symbols):
                early_bin = raw_bits[2 * i]     if 2 * i     < len(raw_bits) else -1
                late_bin  = raw_bits[2 * i + 1] if 2 * i + 1 < len(raw_bits) else -1

                if early_bin != -1 and late_bin == -1:
                    # Photon in early slot only → bit 1
                    detected_indices.append(i)
                    detected_bits.append(1)
                elif early_bin == -1 and late_bin != -1:
                    # Photon in late slot only → bit 0
                    detected_indices.append(i)
                    detected_bits.append(0)
                elif early_bin != -1 and late_bin != -1:
                    # Both slots fired → candidate decoy (not added to data)
                    decoy_candidates.append(i)
                # else: neither → photon lost, discard

            # Accumulate into the run-level buffers
            offset = len(self.received_indices)
            self.received_indices.extend(detected_indices)
            self.received_bits.extend(detected_bits)
            
            self.start_time = self.owner.timeline.now()

            # send message that we got photons
            message = COWMessage(COWMsgType.RECEIVED_QUBITS, self.another.name, indices=detected_indices, bits=detected_bits)
            self.owner.send_message(self.another.owner.name, message)

    def check_visibility(self, decoy_indices: List[int], detected_indices: List[int], detected_bits: List[int],) -> float:
        """Estimate the monitoring-line visibility for the decoy sequences.

        In the COW protocol, consecutive coherent pulses (decoy symbols)
        should interfere constructively at Bob's monitoring detector,
        yielding high visibility.  A reduction in visibility signals
        eavesdropping.

        This implementation uses a simplified model: visibility is
        estimated as the ratio of decoy symbols actually detected (in
        either time bin) to the total number of decoy symbols sent.
        A full simulation would route decoy photons to an interferometer
        and measure fringe contrast directly.

        Args:
            decoy_indices    (list[int]): Alice's decoy-symbol positions.
            detected_indices (list[int]): Bob's detected-symbol positions.
            detected_bits    (list[int]): Bob's inferred bits (unused here).

        Returns:
            float: estimated visibility ∈ [0, 1].
                   Returns 1.0 when no decoys were sent.
        """
        if not decoy_indices:
            return 1.0

        decoy_set     = set(decoy_indices)
        detected_set  = set(detected_indices)
        detected_decoys = decoy_set & detected_set

        visibility = len(detected_decoys) / len(decoy_set)
        return visibility
    
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

                # Initialise reception buffers
                self.received_indices = []
                self.received_bits    = []

                # Set detector to direct-detection (passive, no basis switching)
                self.set_measure_basis_list()
        
                # schedule end_photon_pulse()
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + round(self.light_time * 1e12) - 1, process)
                self.owner.timeline.schedule(event)

            elif msg.msg_type is COWMsgType.RECEIVED_QUBITS:  # (Current node is Alice): can send basis
                log.logger.debug(self.name + " received RECEIVED_QUBITS message")
                # Buffer Bob's detections for key extraction after decoy reveal
                self._pending_bob_indices = list(msg.indices)
                self._pending_bob_bits    = list(msg.bits)

                # Reveal which symbols were decoys
                decoy_pos = (self.decoy_positions.pop(0) if self.decoy_positions else [])
                message = COWMessage(COWMsgType.DECOY_POSITIONS, self.another.name, decoy_indices=decoy_pos)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.DECOY_POSITIONS:  # (Current node is Bob): compare bases
                log.logger.debug(self.name + " received DECOY_POSITIONS")
                decoy_indices = msg.decoy_indices
                decoy_set = set(decoy_indices)

                vis = self.check_visibility(decoy_indices, self.received_indices, self.received_bits)

                self.visibility.append(vis)
                log.logger.info(self.name + f" COW monitoring visibility = {vis:.4f} "f"(threshold={self.VISIBILITY_THRESHOLD})")

                if vis < self.VISIBILITY_THRESHOLD:
                log.logger.warning(self.name + " COW visibility below threshold — possible eavesdropping!")

                # Sift: keep only non-decoy detected symbols
                sifted_indices: List[int] = []
                sifted_bits:    List[int] = []
                for i, idx in enumerate(self.received_indices):
                    if idx not in decoy_set:
                        sifted_indices.append(idx)
                        sifted_bits.append(self.received_bits[i])
                        self.key_bits.append(self.received_bits[i])

                # Send sifted (non-decoy) indices to Alice for key alignment
                message = COWMessage(COWMsgType.SIFTED_INDICES, self.another.name, indices=sifted_indices)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.SIFTED_INDICES:  # (Current node is Alice): create key from matching indices
                log.logger.debug(self.name + " received SIFTED_INDICES")
                # parse matching indices
                sifted_set = set(msg.indices)

                # Retrieve the bit list emitted in the most recent burst
                bit_list = (self.bit_lists.pop(0) if self.bit_lists else [])

                for idx in msg.indices:
                if idx < len(bit_list):
                    self.key_bits.append(bit_list[idx])

                # check if key long enough. If it is, truncate if necessary and call cascade
                if len(self.key_bits) >= self.key_lengths[0]:
                    throughput = self.key_lengths[0] * 1e12 / max(self.owner.timeline.now() - self.last_key_time, 1)

                    while len(self.key_bits) >= self.key_lengths[0] and self.keys_left_list[0] > 0:
                        log.logger.info(self.name + " generated a valid key")
                        self.sifted_bits_length.append(len(self.key_bits))
                        self.set_key()  # convert from binary list to int
                        self._pop(info=self.key)
                        self.another.set_key()
                        self.another._pop(info=self.another.key)  # TODO: why access another node?

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

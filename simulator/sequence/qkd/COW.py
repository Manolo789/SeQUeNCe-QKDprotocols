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

        log.logger.info(self.name + f" generating keys, keylen={length}, keynum={key_num}")

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
            self.basis_lists = []
            self.another.basis_lists = []
            self.bit_lists = []
            self.another.bit_lists = []
            self.key_bits = []
            self.another.key_bits = []
            self.latency = 0
            self.another.latency = 0

            self.working = True
            self.another.working = True

            ls = self.owner.components[self.ls_name]
            self.ls_freq = ls.frequency

            # calculate light time based on key length
            self.light_time = self.key_lengths[0] / (self.ls_freq * ls.mean_photon_num)

            # send message that photon pulse is beginning, then send bits
            self.start_time = int(self.owner.timeline.now()) + round(self.owner.cchannels[self.another.owner.name].delay)
            message = COWMessage(COWMsgType.BEGIN_PHOTON_PULSE, self.another.name,
                                  frequency=self.ls_freq, light_time=self.light_time,
                                  start_time=self.start_time, wavelength=ls.wavelength)
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

            # generate bit list
            num_pulses = round(self.light_time * self.ls_freq)
            bit_list = numpy.random.choice([0, 1], num_pulses)

            # control hardware
            lightsource = self.owner.components[self.ls_name]
            encoding_type = lightsource.encoding_type
            state_list = []
            for i, bit in enumerate(bit_list):
                state = (encoding_type["bases"][bit])[bit]
                state_list.append(state)
            lightsource.emit(state_list)

            self.basis_lists.append(bit_list)
            self.bit_lists.append(bit_list)

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
        """Method to set measurement basis list."""

        log.logger.debug(self.name + " setting measurement basis")

        num_pulses = int(self.light_time * self.ls_freq)
        basis_list = numpy.random.choice([0, 1], num_pulses)
        self.basis_lists.append(basis_list)
        self.owner.components[self.qsd_name].set_basis_list(basis_list, self.start_time, self.ls_freq)

    def end_photon_pulse(self) -> None:
        """Method to process sent qubits."""

        log.logger.debug(self.name + " ending photon pulse")

        if self.working and self.owner.timeline.now() < self.end_run_times[0]:
            # get bits
            self.bit_lists.append(self.owner.get_bits(self.light_time, self.start_time, self.ls_freq, self.qsd_name))
            self.start_time = self.owner.timeline.now()
            # set bases for measurement
            self.set_measure_basis_list()

            # schedule another if necessary
            if self.owner.timeline.now() + self.light_time * 1e12 - 1 < self.end_run_times[0]:
                # schedule another
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + int(round(self.light_time * 1e12) - 1), process)
                self.owner.timeline.schedule(event)

            # send message that we got photons
            message = COWMessage(COWMsgType.RECEIVED_QUBITS, self.another.name)
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

                log.logger.debug(self.name + f" received BEGIN_PHOTON_PULSE, ls_freq={self.ls_freq}, light_time={self.light_time}")

                self.start_time = int(msg.start_time) + self.owner.qchannels[src].delay

                # generate and set basis list
                self.set_measure_basis_list()

                # schedule end_photon_pulse()
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + round(self.light_time * 1e12) - 1, process)
                self.owner.timeline.schedule(event)

            elif msg.msg_type is COWMsgType.RECEIVED_QUBITS:  # (Current node is Alice): can send basis
                log.logger.debug(self.name + " received RECEIVED_QUBITS message")
                bases = self.basis_lists.pop(0)
                message = COWMessage(COWMsgType.BASIS_LIST, self.another.name, bases=bases)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.BASIS_LIST:  # (Current node is Bob): compare bases
                log.logger.debug(self.name + " received BASIS_LIST message")
                # parse alice basis list
                basis_list_alice = msg.bases

                # compare own basis with basis message and create list of matching indices
                indices = []
                basis_list = self.basis_lists.pop(0)
                bits = self.bit_lists.pop(0)
                for i, b in enumerate(basis_list_alice):
                    if bits[i] != -1 and basis_list[i] == bits[i]:
                        indices.append(i)
                        self.key_bits.append(bits[i])

                # send to Alice list of matching indices
                message = COWMessage(COWMsgType.MATCHING_INDICES, self.another.name, indices=indices)
                self.owner.send_message(self.another.owner.name, message)

            elif msg.msg_type is COWMsgType.MATCHING_INDICES:  # (Current node is Alice): create key from matching indices
                log.logger.debug(self.name + " received MATCHING_INDICES message")
                # parse matching indices
                indices = msg.indices

                bits = self.bit_lists.pop(0)

                # set key equal to bits at received indices
                for i in indices:
                    self.key_bits.append(bits[i])

                # check if key long enough. If it is, truncate if necessary and call cascade
                if len(self.key_bits) >= self.key_lengths[0]:
                    throughput = self.key_lengths[0] * 1e12 / (self.owner.timeline.now() - self.last_key_time)

                    while len(self.key_bits) >= self.key_lengths[0] and self.keys_left_list[0] > 0:
                        log.logger.info(self.name + " generated a valid key")
                        self.sifted_bits_length.append(len(self.key_bits)) ### Editado
                        self.set_key()  # convert from binary list to int
                        self._pop(info=self.key)
                        self.another.set_key()
                        self.another._pop(info=self.another.key)  # TODO: why access another node?

                        # for metrics
                        if self.latency == 0:
                            self.latency = (self.owner.timeline.now() - self.last_key_time) * 1e-12

                        self.throughputs.append(throughput)

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

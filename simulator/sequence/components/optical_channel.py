"""Models for simulation of optical fiber channels.

This module introduces the abstract OpticalChannel class for general optical fibers.
It also defines the QuantumChannel class for transmission of qubits/photons and the ClassicalChannel class for transmission of classical control messages.
OpticalChannels must be attached to nodes on both ends.


NOISE FIX: Added channel phase decoherence for time_bin_cow encoding.
Each photon traversing the fiber accumulates a random phase drawn from
N(0, σ²) where σ depends on the fiber length.  This phase is stored
in photon.channel_phase and is used by the Michelson interferometer
to compute the relative phase between two consecutive pulses.

"""

import heapq as hq
import math
import gmpy2
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node, EveNode
    from ..components.photon import Photon
    from ..message import Message

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log
from ..constants import SPEED_OF_LIGHT, MICROSECOND, SECOND, EPSILON

gmpy2.get_context().precision = 80 # 80 bits ~ 24 decimal digits ~ sufficient for 10,000 years of ps timing 
EPSILON_MPFR = gmpy2.mpfr(EPSILON)
PS_PER_SECOND = gmpy2.mpz(SECOND)


class OpticalChannel(Entity):
    """Parent class for optical fibers.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (float): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: float,
                 polarization_fidelity: float, light_speed: float):
        """Constructor for abstract Optical Channel class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (float): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
            light_speed (float): speed of light within the fiber (in m/ps).
        """
        log.logger.info(f"Create channel {name}")

        Entity.__init__(self, name, timeline)
        self.sender = None
        self.receiver = None
        self.attenuation = attenuation
        self.distance = distance  # (measured in m)
        self.polarization_fidelity = polarization_fidelity
        self.light_speed = light_speed  # used for photon timing calculations (measured in m/ps)

    def init(self) -> None:
        pass

    def set_distance(self, distance: float) -> None:
        self.distance = distance


class QuantumChannel(OpticalChannel):
    """Optical channel for transmission of photons/qubits.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (float): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
        loss (float): loss rate for transmitted photons (determined by attenuation).
        delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
        frequency (float): maximum frequency of qubit transmission (in Hz).
        phase_noise_coefficient (float): standard deviation of phase noise per
            sqrt(meter) of fiber (in rad/sqrt(m)).  The total phase std-dev
            for a channel of length L is σ_φ = phase_noise_coefficient × √L.
            Default 0.0 (no channel phase noise).
            Typical value: 0.005–0.02 rad/√m for standard telecom fiber.

    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: float,
                 polarization_fidelity: float = 1.0, light_speed: float = SPEED_OF_LIGHT, frequency: float = 8e6, phase_noise_coefficient: float = 0.0,
                 loss: "float | None" = None):
        """Constructor for Quantum Channel class.

        Args:
            name (str): name of the quantum channel instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (float): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit (default 1).
            light_speed (float): speed of light within the fiber (in m/ps).
            delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
            loss (float): loss rate for transmitted photons (determined by attenuation).
            frequency (float): maximum frequency of qubit transmission (in Hz) (default 8e7).
            phase_noise_coefficient (float): phase noise per sqrt(m) (default 0.0).

        """

        super().__init__(name, timeline, attenuation, distance, polarization_fidelity, light_speed)
        self.delay: int = -1
        self.loss: float = 1
        self.frequency: float = frequency  # maximum frequency for sending qubits (measured in Hz)
        self.send_bins: list = []
        self.phase_noise_coefficient: float = phase_noise_coefficient
        self._loss_spec = loss  # None → default formula, float → fixed

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        self.delay = round(self.distance / self.light_speed)
        if self._loss_spec is None:
            self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)
        else:
            self.loss = float(self._loss_spec)
        

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the quantum channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending qubits.
            receiver (str): name of node receiving qubits.
        """

        log.logger.info(f"Set {sender.name}, {receiver} as ends of quantum channel {self.name}")
        self.sender = sender
        self.receiver = receiver
        sender.assign_qchannel(self, receiver)
        
    def _apply_channel_noise(self, qubit: "Photon") -> None:
        """Apply channel-level noise to a photon being transmitted.

        For polarization encoding: existing random_noise (flips state).
        For time_bin_cow encoding: random phase noise that degrades
        coherence between consecutive pulses in the Michelson
        interferometer on the monitoring line.

        The phase noise model follows a Wiener process in fiber:
            σ_φ = phase_noise_coefficient × √(distance)

        This phase is stored in photon.channel_phase and is read by
        MichelsonInterferometer._interfere() to compute the interference
        visibility.

        Args:
            qubit (Photon): photon being transmitted.
        """
        rng = self.sender.get_generator()

        if qubit.encoding_type["name"] == "polarization":
            if rng.random() > self.polarization_fidelity:
                qubit.random_noise(self.get_generator())

        elif qubit.encoding_type["name"] == "time_bin_cow":
            # Channel phase decoherence: each photon accumulates a random
            # phase proportional to sqrt(fiber length).
            # This does NOT affect the data line (time-of-arrival is
            # phase-independent), but it DOES reduce the visibility
            # measured by the Michelson interferometer.
            if self.phase_noise_coefficient > 0 and self.distance > 0:
                sigma_phi = self.phase_noise_coefficient * math.sqrt(self.distance)
                qubit.channel_phase = float(rng.normal(0.0, sigma_phi))



    def transmit(self, qubit: "Photon", source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info("{} send qubit with state {} to {} by Channel {}".format(
                        self.sender.name, qubit.quantum_state, self.receiver, self.name))

        assert self.delay >= 0 and self.loss <= 1, f"QuantumChannel init() function has not been run for {self.name}"
        assert source == self.sender

        # remove lowest time bin
        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = self.timebin_to_time(time_bin, self.frequency)
            assert time == self.timeline.now(), f"qc {self.name} transmit method called at invalid time"

        # check if photon state using Fock representation
        if qubit.encoding_type["name"] == "fock":
            key = qubit.quantum_state  # if using Fock representation, the `quantum_state` field is the state key.
            # apply loss channel on photonic statex
            self.timeline.quantum_manager.add_loss(key, self.loss)

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, check if photon kept
        elif (self.sender.get_generator().random() > self.loss) or qubit.is_null:
            if self._receiver_on_other_tl():
                self.timeline.quantum_manager.move_manage_to_server(qubit.quantum_state)

            if qubit.is_null:
                qubit.add_loss(self.loss)
                
            

            # ── Apply channel noise (polarization OR phase decoherence) ──
            self._apply_channel_noise(qubit)


            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, if photon lost, exit
        else:
            pass

    def time_to_timebin(self, time: int, frequency: float) -> int:
        """Convert simulation time to time bin.
           Use the gmpy2.mpfr for high precision floating points.
           The precision is set to 200 bits, equivalent to around 54 significant decimal digits.
           The float in Python is 64 bits,   equivalent to around 16 significant decimal digits.

        Args:
            time (int): simulation time (picoseconds) to convert.
            frequency (float): frequency of the channel.
        Returns:
            int: time bin corresponding to the given simulation time.
        """
        time = gmpy2.mpfr(time)
        frequency = gmpy2.mpfr(frequency)
        time_bin = time * frequency / PS_PER_SECOND
        if time_bin - gmpy2.floor(time_bin) > EPSILON_MPFR:
            time_bin = int(time_bin) + 1       # round to the next time bin
        else:
            time_bin = int(time_bin)
        return time_bin

    def timebin_to_time(self, time_bin: int, frequency: float) -> int:
        """Convert time bin to simulation time (picoseconds).
           Use the gmpy2.mpz  for high precision integers.
           Use the gmpy2.mpfr for high precision floating points.

        Args:
            time_bin (int): time bin to convert.
            frequency (float): frequency of the channel.

        Returns:
            int: simulation time (picoseconds) corresponding to the given time bin.
        """
        time_bin = gmpy2.mpz(time_bin)
        frequency = gmpy2.mpfr(frequency)
        time = gmpy2.mpfr(time_bin * PS_PER_SECOND) / frequency
        return int(time)

    def schedule_transmit(self, min_time: int) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by a frequency of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """
        min_time = max(min_time, self.timeline.now())
        time_bin = self.time_to_timebin(min_time, self.frequency)

        # find earliest available time bin
        while time_bin in self.send_bins:
            time_bin += 1
        hq.heappush(self.send_bins, time_bin)

        time = self.timebin_to_time(time_bin, self.frequency)
        return time

    def _receiver_on_other_tl(self) -> bool:
        return self.timeline.get_entity_by_name(self.receiver) is None

class EveQuantumChannel(QuantumChannel):
    """Canal quântico com intercepção transparente por Eve.

    Eve é inserida entre Alice e Bob **sem que nenhum deles saiba**.
    O setup da simulação permanece idêntico ao caso sem Eve:

        qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, ...)
        qc0.set_ends(alice, bob.name)   # ← mesmo código que sem Eve

    Internamente, o canal divide-se em dois segmentos:
        alice ──[seg1]──► eve ──[seg2]──► bob

    O canal `seg2` (Eve → Bob) é criado automaticamente no `init()`.

    Attributes:
        eve_node (EveNode): nó espiã inserido no canal.
        eve_position (float): posição fracionária de Eve ao longo do
            canal (0 = junto de Alice, 1 = junto de Bob). Default 0.5.
        _seg2 (QuantumChannel): segmento interno Eve → Bob, criado em
            `init()` após `set_ends` definir o receptor final.
    """

    def __init__(self, name: str, timeline: "Timeline", eve_node: "EveNode", attenuation: float, distance: float, 
        polarization_fidelity: float = 1.0, light_speed: float = SPEED_OF_LIGHT, frequency: float = 8e6, eve_position: float = 0.5, phase_noise_coefficient: float = 0.0, 
        loss: "float | None" = None) -> None:
        """
        Args:
            name:                 nome do canal.
            timeline:             timeline da simulação.
            eve_node:             nó EveNode a inserir no meio do canal.
            attenuation:          atenuação da fibra em dB/m.
            distance:             distância total Alice→Bob em metros.
            polarization_fidelity: fidelidade de polarização (0–1).
            light_speed:          velocidade da luz na fibra em m/ps.
            frequency:            frequência máxima de transmissão em Hz.
            eve_position:         posição fracionária de Eve (0–1).
        """
        dist_seg1 = distance * eve_position
        # O segmento 1 (Alice→Eve) é o próprio canal — herda QuantumChannel
        super().__init__(
            name, timeline,
            attenuation=attenuation,
            distance=dist_seg1,
            polarization_fidelity=polarization_fidelity,
            light_speed=light_speed,
            frequency=frequency,
            phase_noise_coefficient=phase_noise_coefficient,
            loss=loss,
        )
        self.eve_node: "EveNode" = eve_node
        self.eve_position: float = eve_position
        self._total_distance: float = distance
        self._seg2: Optional[QuantumChannel] = None   # criado em init()

    # ── QuantumChannel interface ──────────────────────────────────────────

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Registra Alice como emissora e intercepta o canal para Eve.

        O canal é registrado em Alice sob o nome do receptor final (Bob),
        de modo que `alice.send_qubit('bob', photon)` use este canal.
        Eve permanece invisível para o protocolo.

        Args:
            sender:   nó de Alice.
            receiver: nome do nó de Bob (receptor final).
        """
        # ── Segmento 1: Alice → Eve ───────────────────────────────────
        # Registra este canal em Alice com a chave 'bob' (não 'eve').
        # Assim alice.send_qubit('bob') usa este canal sem saber de Eve.
        self.sender   = sender
        self.receiver = receiver
        sender.assign_qchannel(self, receiver)   # alice.qchannels['bob'] = self

        # ── Segmento 2: Eve → Bob ─────────────────────────────────────
        # Criado aqui (set_ends), não em init(), para não modificar
        # timeline.entities durante a iteração de Timeline.init().
        dist_seg2 = self._total_distance * (1.0 - self.eve_position)
        seg2_name = f"{self.name}.seg2"
        self._seg2 = QuantumChannel(
            seg2_name,
            self.timeline,
            attenuation=self.attenuation,
            distance=dist_seg2,
            polarization_fidelity=self.polarization_fidelity,
            light_speed=self.light_speed,
            frequency=self.frequency,
            phase_noise_coefficient=self.phase_noise_coefficient,
            loss=self._loss_spec,
        )
        # Registra _seg2 em Eve com a chave 'bob'.
        # Eve chamará eve.send_qubit('bob') após intercepção.
        self._seg2.set_ends(self.eve_node, self.receiver)

        # Define o destino de retransmissão de Eve
        self.eve_node.destination = self.receiver

    # ── init: nada a fazer — Timeline.init() cuida de _seg2 ──────────────

    def init(self) -> None:
        """Inicializa ambos os segmentos e conecta Eve→Bob.

        Chamado por Timeline.init() após set_ends ter sido executado.
        `self.receiver` já contém o nome do receptor final (Bob).
        """
        # Inicializa o segmento 1 (Alice → Eve)
        super().init()
        
    # ── transmit: redireciona fótons para Eve ─────────────────────────────

    def transmit(self, qubit: "Photon", source: "Node") -> None:
        """Transmite um fóton pelo segmento Alice→Eve.

        Aplica perda e ruído de polarização do segmento 1, depois agenda
        a entrega a Eve (não a Bob). Eve decidirá se intercepta ou
        encaminha via `_seg2`.

        O método é idêntico ao `QuantumChannel.transmit()`, exceto que
        o receptor agendado é `eve_node` em vez de `self.receiver`.
        """
        assert self.delay >= 0 and self.loss <= 1, \
            f"EveQuantumChannel.init() não foi executado para {self.name}"
        assert source == self.sender

        import heapq as hq
        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = self.timebin_to_time(time_bin, self.frequency)
            assert time == self.timeline.now()

        if qubit.encoding_type["name"] == "fock":
            key = qubit.quantum_state
            self.timeline.quantum_manager.add_loss(key, self.loss)
            self._schedule_to_eve(qubit, source)

        elif (self.sender.get_generator().random() > self.loss) or qubit.is_null:
            if qubit.is_null:
                qubit.add_loss(self.loss)

            # Apply channel noise for segment 1 (Alice → Eve)
            self._apply_channel_noise(qubit)


            self._schedule_to_eve(qubit, source)
        # fóton perdido: não agenda nada

    def _schedule_to_eve(self, qubit: "Photon", source: "Node") -> None:
        """Agenda entrega do fóton a Eve após o atraso do segmento 1."""
        future_time = self.timeline.now() + self.delay
        process = Process(self.eve_node.name, "receive_qubit",
                          [source.name, qubit])
        event = Event(future_time, process)
        self.timeline.schedule(event)


class ClassicalChannel(OpticalChannel):
    """Optical channel for transmission of classical messages.

    Classical message transmission is assumed to be lossless.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        distance (float): length of the fiber (in m).
        delay (float): delay (in ps) of message transmission (default distance / light_speed).
    """

    def __init__(self, name: str, timeline: "Timeline", distance: float, delay: int = -1):
        """Constructor for Classical Channel class.

        Args:
            name (str): name of the classical channel instance.
            timeline (Timeline): simulation timeline.
            distance (float): length of the fiber (in m).
            delay (float): delay (in ps) of message transmission (default distance / light_speed).
        """

        super().__init__(name, timeline, 0, distance, 0, SPEED_OF_LIGHT)
        if delay == -1:
            self.delay = round(distance / self.light_speed + 10 * MICROSECOND)
        else:
            self.delay = round(delay)

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the classical channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending classical messages.
            receiver (str): name of node receiving classical messages.
        """

        log.logger.info(f"Set {sender.name}, {receiver} as ends of classical channel {self.name}")
        self.sender = sender
        self.receiver = receiver
        sender.assign_cchannel(self, receiver)

    def transmit(self, message: "Message", source: "Node", priority: int) -> None:
        """Method to transmit classical messages.

        Args:
            message (Message): message to be transmitted.
            source (Node): node sending the message.
            priority (int): priority of transmitted message (to resolve message reception conflicts).

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info(f"{self.sender.name} send message {message} to {self.receiver} by Channel {self.name}")
        assert source == self.sender

        future_time = round(self.timeline.now() + int(self.delay))
        process = Process(self.receiver, "receive_message", [source.name, message])
        event = Event(future_time, process, priority)
        self.timeline.schedule(event)

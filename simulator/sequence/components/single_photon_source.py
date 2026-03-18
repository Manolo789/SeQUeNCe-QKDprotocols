"""Single-photon source for QKD simulations.

This module provides :class:`SinglePhotonSource`, a subclass of
:class:`~sequence.components.light_source.LightSource` that emits
**exactly one photon per time slot** instead of drawing from a Poisson
distribution.

This models an ideal single-photon source (SPS) — e.g. a quantum dot,
nitrogen-vacancy centre, or heralded SPDC source — where multi-photon
events are suppressed by construction.


"""

from numpy import multiply

from sequence.components.light_source import LightSource
from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.utils.encoding import polarization
from sequence.utils.encoding_cow import VACUUM_STATE, time_bin_cow
from sequence.utils import log


class SinglePhotonSource(LightSource):
    """Ideal single-photon source for BB84 / B92.

    Emits exactly **one photon** per time slot, eliminating multi-photon
    pulses entirely.  The ``mean_photon_num`` parameter is ignored for
    emission but kept for compatibility with protocol-level calculations
    (e.g. ``light_time`` estimation).

    All other behaviour (wavelength, phase error, encoding type) is
    inherited from :class:`LightSource`.
    """

    def __init__(self, name, timeline, frequency=8e7, wavelength=1550,
                 bandwidth=0, mean_photon_num=1.0, encoding_type=polarization,
                 phase_error=0):
        super().__init__(name, timeline, frequency, wavelength, bandwidth,
                         mean_photon_num, encoding_type, phase_error)

    def emit(self, state_list) -> None:
        """Emit exactly one photon per state in *state_list*.

        Unlike the parent class, no Poisson draw is performed.  Each
        entry in ``state_list`` produces exactly one photon object.

        Args:
            state_list: list of quantum states (complex coefficient tuples).
        """
        log.logger.info(f"{self.name} (SPS) emitting {len(state_list)} photons")

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):
            # Optional phase error (same as parent)
            if self.get_generator().random() < self.phase_error:
                state = multiply([1, -1], state)

            # Deterministic: exactly 1 photon
            wavelength = (self.linewidth * self.get_generator().standard_normal()
                          + self.wavelength) if self.linewidth > 0 else self.wavelength

            new_photon = Photon(str(i), self.timeline,
                                wavelength=wavelength,
                                location=self.owner,
                                encoding_type=self.encoding_type,
                                quantum_state=state)
            process = Process(self._receivers[0], "get", [new_photon])
            event = Event(time, process)
            self.timeline.schedule(event)
            self.photon_counter += 1

            time += period

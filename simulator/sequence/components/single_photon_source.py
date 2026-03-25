"""
=====================================================================
Single-photon source (SPS) model in the SeQUeNCe simulator -- License
=====================================================================

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

=====================================================================

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

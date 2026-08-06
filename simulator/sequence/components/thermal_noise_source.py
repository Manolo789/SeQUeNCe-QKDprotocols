"""
======================================================================
Thermal background photon source for the SeQUeNCe simulator -- License
======================================================================

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

======================================================================

"""

import math
from typing import TYPE_CHECKING

from numpy.random import default_rng

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization
from ..utils.encoding_cow import time_bin_cow
from QCLoss.sky_radiance import n_background

_h = 6.62607015e-34   # J·s
_c = 2.99792458e8     # m/s


class ThermalNoiseSource(Entity):
    """Source of background thermal photons for an FSO link.

    Emits photons in a random state, modelling the sky radiance collected
    through the receiver aperture.

    Attributes:
        n_B (float): background photons per mode [dimensionless]
        frequency (float): clock frequency [Hz]
        encoding_type (dict): encoding scheme of the emitted photons
        active (bool): switches the source on and off
    """

    def __init__(self, name: str, timeline: "Timeline", n_B: float, frequency: float, encoding_type: dict = None, detection_gate: float = None, seed: int = None) -> None:
        """Build a seeded background source and derive its emission rate.

        The source keeps its OWN seeded generator because this Entity is
        not registered on a Node: with ``self.owner is None``,
        ``Entity.get_generator()`` would fall back to an unseeded
        ``default_rng()`` on every call, which both breaks reproducibility
        (even with alice/bob seeds fixed) and allocates one Generator per
        background photon.

        The emission rate implements the temporal gating that the
        simulator's Detector lacks. n_B (Pirandola PRR 3, 023130, Eq. 32)
        is defined per temporal MODE of duration ``detection_gate``, so the
        raw sky rate is n_B/detection_gate; a QKD receiver only accepts the
        gate (~1 ns) centred on each pulse (period 1/frequency, ~125 ns).
        Applying that duty cycle here gives
        lambda = (n_B/detection_gate) * (detection_gate * frequency)
               = n_B * frequency  [DETECTABLE photons/s].
        Injecting the raw rate instead would overestimate the counts by the
        period/gate ratio (~125x in the base scenario), saturate the
        detector dead time (QBER ~ 0.3) and inflate the timeline event
        count by the same factor.

        Args:
            name: entity name.
            timeline: timeline that owns this entity.
            n_B: background photons per detection mode.
            frequency: clock frequency of the link [Hz].
            encoding_type: encoding scheme; polarization when None.
            detection_gate: temporal acceptance window [s].
            seed: seed of the source RNG, for reproducibility.
        """
        Entity.__init__(self, name, timeline)

        if encoding_type is None:
            encoding_type = polarization

        self._rng = default_rng(seed)

        self.n_B = n_B
        self.frequency = frequency
        self.encoding_type = encoding_type
        self.detection_gate = detection_gate
        self.active = True
        self._recompute_rate()

    def _recompute_rate(self) -> None:
        """(Re)compute the effective rate of DETECTABLE background photons.

        See :meth:`__init__` for why the duty cycle is applied here.
        """
        if self.frequency is not None and self.frequency > 0:
            # Gated model: n_B photons per gate x frequency gates/s.
            self._arrival_rate = self.n_B * self.frequency
        elif self.detection_gate is not None and self.detection_gate > 0:
            # No reference frequency: raw background, ungated.
            import warnings
            warnings.warn(
                "ThermalNoiseSource without frequency: using the raw rate "
                "n_B/detection_gate (no temporal gate). The simulator's "
                "detector has no gate, so the background counts will be "
                "overestimated by the period/gate ratio.",
                RuntimeWarning)
            self._arrival_rate = self.n_B / self.detection_gate
        else:
            self._arrival_rate = 0.0
            
    def get_generator(self):
        """Return the seeded generator: the owner node's when there is one.

        Returns:
            np.random.Generator: RNG used by this source.
        """
        if hasattr(self.owner, "get_generator"):
            return self.owner.get_generator()
        return self._rng

    def init(self) -> None:
        """Schedule the first background-photon emission event."""
        if self._arrival_rate > 0 and self.active:
            self._schedule_next()

    def _schedule_next(self) -> None:
        """Schedule the next background photon (Poisson arrivals)."""
        # Exponential interval: E[T] = 1/lambda (in seconds).
        dt_s = self.get_generator().exponential(1.0 / self._arrival_rate)
        dt_ps = int(dt_s * 1e12)

        t_next = self.timeline.now() + max(dt_ps, 1)
        process = Process(self, "_emit", [])
        event = Event(t_next, process)
        self.timeline.schedule(event)

    def _emit(self) -> None:
        """Create a background photon in a random state and send it on."""
        if not self.active:
            return

        state  = self._random_state()
        photon = Photon(
            name          = "bg_photon",
            timeline      = self.timeline,
            encoding_type = self.encoding_type,
            quantum_state = state,
        )
        # Thermal light has a random phase, hence no coherence with the
        # signal pulses: accidental (signal, background) pairs must route
        # 50/50 in the Michelson instead of interfering at channel_phase=0.
        photon.coherent = False
        # A real photon, so it goes through the Detector efficiency.
        self._receivers[0].get(photon)

        # Schedule the next photon.
        self._schedule_next()

    def _random_state(self) -> tuple:
        """Random quantum state modelling an incoherent background.

        Polarisation: random (equatorial) state on the Bloch sphere.
        time_bin / time_bin_cow: early or late with equal probability.

        Returns:
            tuple: the quantum state accepted by ``Photon.__init__``.
        """
        rng = self.get_generator()

        if self.encoding_type["name"] == "polarization":
            # Polarisation angle uniform in [0, pi).
            theta = rng.uniform(0, math.pi)
            return (complex(math.cos(theta)), complex(math.sin(theta)))

        elif self.encoding_type["name"] in ("time_bin", "time_bin_cow"):
            # Arrival in the early (0) or late (1) bin with p=0.5.
            # Photon.__init__ requires a tuple, while time_bin_cow entries
            # are np.ndarray, hence the explicit tuple() conversion.
            if rng.random() < 0.5:
                return tuple(time_bin_cow["early"]) if "cow" in self.encoding_type["name"] \
                        else (complex(1), complex(0))
            else:
                return tuple(time_bin_cow["late"]) if "cow" in self.encoding_type["name"] \
                        else (complex(0), complex(1))
        else:
            # Generic fallback: uniform superposition.
            return (complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)))

    def set_n_B(self, n_B: float) -> None:
        """Update n_B at run time, e.g. on a night-to-day transition.

        Args:
            n_B: new number of background photons per detection mode.
        """
        self.n_B = n_B
        self._recompute_rate()   # same rule as the constructor



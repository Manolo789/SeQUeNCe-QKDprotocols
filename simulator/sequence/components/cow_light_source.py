"""
============================================================================
Light source model for the COW protocol in the SeQUeNCe simulator -- License
============================================================================

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

============================================================================

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from .light_source import LightSource
from .photon import Photon
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding_cow import VACUUM_STATE, time_bin_cow


class COWLightSource(LightSource):
    """Weak-coherent-pulse source for the COW QKD protocol.

    Attributes:
        extinction_ratio (float): intensity modulator extinction ratio
            (linear, not dB).  Default 0 means perfect extinction (no
            leakage in vacuum slots).
            Typical value: 100–1000 (20–30 dB).
            Set to e.g. 100 for ER = 20 dB, or 1000 for ER = 30 dB.
    """

    def __init__(
        self,
        name: str,
        timeline,
        frequency: float = 8e6,
        wavelength: float = 780,
        bandwidth: float = 0,
        mean_photon_num: float = 0.5,
        encoding_type: dict = None,
        phase_error: float = 0,
        extinction_ratio: float = 0,
    ) -> None:
        """Construct a COWLightSource.

        Args:
            name (str): component label.
            timeline (Timeline): simulation timeline.
            frequency (float): clock frequency in Hz (default 434 MHz as in
                Stucki et al.).
            wavelength (float): photon wavelength in nm (default 1550).
            bandwidth (float): linewidth standard deviation in nm (default 0).
            mean_photon_num (float): mean photon number μ per pulse slot
                (default 0.5, as used in the COW experiment).
            encoding_type (dict): encoding dictionary; defaults to
                ``time_bin_cow``.
            phase_error (float): probability of a π phase flip per pulse
                (default 0).
            extinction_ratio (float): IM extinction ratio (linear).
                0 = perfect extinction.  100 = 20 dB.  1000 = 30 dB.
        """
        if encoding_type is None:
            encoding_type = time_bin_cow
        super().__init__(
            name,
            timeline,
            frequency=frequency,
            wavelength=wavelength,
            bandwidth=bandwidth,
            mean_photon_num=mean_photon_num,
            encoding_type=encoding_type,
            phase_error=phase_error,
        )
        self.extinction_ratio = extinction_ratio


    def emit(self, state_list: list) -> None:
        """Emit photons for each state in *state_list*.

        For each entry:
        * If the entry is ``VACUUM_STATE`` (``None``): advance the clock by
          one slot period and emit nothing — deterministic vacuum.
        * Otherwise: draw ``Poisson(mean_photon_num)`` photons and schedule
          them with the given quantum state, exactly as the parent class does.

        Args:
            state_list (list): list of quantum states or ``VACUUM_STATE``
                sentinels, one per time slot.  Produced by
                :func:`~sequence.utils.encoding_cow.build_cow_state_list`.
        """
        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))   # slot period in ps
        rng    = self.get_generator()
        
        # Compute leakage mean photon number
        mu_leak = 0.0
        if self.extinction_ratio > 0:
            mu_leak = self.mean_photon_num / self.extinction_ratio

        # Separate vacuum and non-vacuum slots
        # For vacuum slots with leakage, we still need to emit (weak) photons
        vacuum_with_leak: list[tuple[int, object]] = []
        non_vacuum: list[tuple[int, object]] = []

        for i, s in enumerate(state_list):
            if s is VACUUM_STATE:
                if mu_leak > 0:
                    # Use EARLY_STATE as the leakage state (arbitrary —
                    # leakage has random phase in practice, but for the
                    # time-of-arrival measurement it just creates a click
                    # in the wrong bin)
                    vacuum_with_leak.append((i, time_bin_cow["early"]))
            else:
                non_vacuum.append((i, s))

        # ── Non-vacuum slots: normal Poisson emission ──
        if non_vacuum:
            n_nv = len(non_vacuum)
            counts = rng.poisson(self.mean_photon_num, size=n_nv)

            for k in np.nonzero(counts)[0]:
                i, state = non_vacuum[k]
                slot_time = time + i * period
                n_photons = int(counts[k])

                if self.phase_error > 0 and rng.random() < self.phase_error:
                    state = tuple(np.multiply([1, -1], state))

                for _ in range(n_photons):
                    wl = (
                        self.linewidth * rng.standard_normal() + self.wavelength
                        if self.linewidth > 0
                        else self.wavelength
                    )
                    photon = Photon(
                        str(i),
                        self.timeline,
                        wavelength=wl,
                        location=self.owner,
                        encoding_type=self.encoding_type,
                        quantum_state=tuple(state),
                    )
                    process = Process(self._receivers[0], "get", [photon])
                    event = Event(slot_time, process)
                    self.timeline.schedule(event)
                    self.photon_counter += 1

        # ── Vacuum slots with leakage: weak Poisson emission ──
        if vacuum_with_leak:
            n_vl = len(vacuum_with_leak)
            leak_counts = rng.poisson(mu_leak, size=n_vl)

            for k in np.nonzero(leak_counts)[0]:
                i, state = vacuum_with_leak[k]
                slot_time = time + i * period
                n_photons = int(leak_counts[k])

                for _ in range(n_photons):
                    wl = (
                        self.linewidth * rng.standard_normal() + self.wavelength
                        if self.linewidth > 0
                        else self.wavelength
                    )
                    photon = Photon(
                        str(i),
                        self.timeline,
                        wavelength=wl,
                        location=self.owner,
                        encoding_type=self.encoding_type,
                        quantum_state=tuple(state),
                    )
                    process = Process(self._receivers[0], "get", [photon])
                    event = Event(slot_time, process)
                    self.timeline.schedule(event)
                    self.photon_counter += 1

"""COW-aware light source.

This module provides :class:`COWLightSource`, a thin subclass of
:class:`~sequence.components.light_source.LightSource` that is able to emit
the *vacuum state* explicitly.

In the standard ``LightSource.emit`` every entry in the state list triggers
a Poisson draw of ``mean_photon_num`` photons.  For the COW protocol the
vacuum slots in a bit-encoded symbol must produce **zero photons with
certainty** (not merely with high probability), because:

* Bit 0  = |μ⟩|0⟩  (pulse then vacuum — second slot is deterministically empty)
* Bit 1  = |0⟩|μ⟩  (vacuum then pulse — first slot is deterministically empty)
* Decoy  = |μ⟩|μ⟩  (both slots carry a weak coherent pulse)

``COWLightSource`` recognises the sentinel ``VACUUM_STATE = None`` exported
from :mod:`sequence.utils.encoding_cow` and skips photon emission for those
slots while still advancing the simulation clock by one slot period.

Usage
-----
Replace ``LightSource`` with ``COWLightSource`` when building a
:class:`~sequence.topology.node.QKDNode` for the COW protocol::

    from sequence.components.cow_light_source import COWLightSource
    from sequence.utils.encoding_cow import time_bin_cow

    ls = COWLightSource("alice.ls", timeline,
                        encoding_type=time_bin_cow,
                        frequency=434e6,
                        mean_photon_num=0.5)
    alice.add_component(ls)
    ls.add_receiver(alice)          # forward photons to node → qchannel

The :meth:`emit` method accepts the flat state list produced by
:func:`~sequence.utils.encoding_cow.build_cow_state_list`.
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

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

    Extends :class:`LightSource` so that the sentinel ``VACUUM_STATE``
    (``None``) in the state list causes a deterministic vacuum slot: the
    clock advances by one period but no photon object is created or
    transmitted.  Every non-``None`` entry is treated exactly as in the
    parent class (Poisson draw with ``mean_photon_num``).

    Attributes:
        (inherits all attributes from LightSource)
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

        # ── Passo 1: separar slots não-vacuum ─────────────────────────────
        # Percorremos state_list uma vez para coletar (índice_slot, estado).
        # Esta lista tem comprimento ≈ N_símbolos (metade de len(state_list))
        # porque metade dos slots são vacuum em média.
        non_vacuum: list[tuple[int, object]] = [
            (i, s) for i, s in enumerate(state_list) if s is not VACUUM_STATE
        ]

        if not non_vacuum:
            return   # burst inteiramente vacuum — nada a emitir

        n_nv = len(non_vacuum)

        # ── Passo 2: draw Poisson vetorizado ──────────────────────────────
        # Uma única chamada C gera todas as contagens.
        # dtype int garante que a comparação counts[k] == 0 seja O(1).
        counts: np.ndarray = rng.poisson(self.mean_photon_num, size=n_nv)

        # ── Passo 3: iterar apenas sobre slots com fótons (≈10% do total) ─
        for k in np.nonzero(counts)[0]:
            i, state = non_vacuum[k]
            slot_time = time + i * period
            n_photons = int(counts[k])

            # Correção de fase opcional (aplicada por slot, não por fóton)
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
                event   = Event(slot_time, process)
                self.timeline.schedule(event)
                self.photon_counter += 1

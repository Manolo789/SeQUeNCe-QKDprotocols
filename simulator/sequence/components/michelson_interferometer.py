"""Model for simulation of a Michelson interferometer for COW QKD monitoring.

This module introduces the MichelsonInterferometer class used on the monitoring
line of the COW (Coherent One-Way) QKD protocol (Stucki et al., Appl. Phys.
Lett. 87, 194108, 2005).

Physical description
--------------------
The interferometer has a single optical input and two output ports (DM1, DM2).
A 50/50 beamsplitter divides each incoming photon between a *short arm* (SA)
and a *long arm* (LA). The long arm introduces an extra optical delay equal to
one clock period τ = 1/f_clock (46 cm of fibre in the original experiment).

Consequence for consecutive pulses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Consider two consecutive non-empty pulses arriving at times t and t + τ
(either a decoy sequence |μ⟩|μ⟩ or a "1→0" bit-sequence transition):

* The SA copy of pulse-(t + τ) arrives at the recombination BS at t + τ.
* The LA copy of pulse-t    arrives at the recombination BS at t + τ.

These two copies overlap spatially and temporally and interfere.  With the
correct relative phase (φ = 0, enforced by laser coherence) only detector
DM1 fires — constructive interference.  Eavesdropping or any decoherence
mixes phases, causing DM2 to fire with finite probability, reducing the
monitored visibility:

    V = (p_DM1 − p_DM2) / (p_DM1 + p_DM2)

Isolated single pulses (no partner arriving one clock period later) are
split randomly 50/50 between the two output ports and do not contribute to
the visibility estimate.

Implementation notes
~~~~~~~~~~~~~~~~~~~~
* The class stores an incoming photon in an internal buffer together with its
  arrival time.  When the next photon arrives within a tolerance window
  centred on (buffer_time + path_difference), two-photon interference is
  applied.  Otherwise the buffered photon is ejected 50/50 after a scheduled
  timeout event.
* ``path_difference`` should be set to 1e12 / f_clock (ps) so that it
  equals the slot period of the COW light source.
* ``phase`` models the thermal phase setting of the real interferometer; the
  paper uses Faraday mirrors, making the device polarisation-insensitive —
  this is implicit here (no polarisation mode tracked).
* ``phase_error`` is a per-event additive Gaussian noise on the phase
  (standard deviation, radians).

Receiver convention
~~~~~~~~~~~~~~~~~~~
    _receivers[0] → DM1  (constructive port, should fire for clean coherence)
    _receivers[1] → DM2  (destructive port, fires when coherence is broken)
"""

from math import cos, sin, pi
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from .photon import Photon

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process


class MichelsonInterferometer(Entity):
    """Michelson interferometer for the COW QKD monitoring line.

    Attributes:
        name (str): label for the instance.
        timeline (Timeline): simulation timeline.
        path_difference (int): optical path difference in ps — must equal one
            clock period (τ = 1e12 / f_clock).
        phase (float): interferometer phase in radians.  0 → constructive
            interference at DM1 (the desired operating point).
        phase_error (float): standard deviation (rad) of per-shot phase noise.
        _photon_buffer (Optional[Photon]): photon currently delayed in the
            long arm.
        _buffer_time (int): timeline time (ps) when the buffered photon
            arrived at the input beamsplitter.
    """

    #: Fraction of ``path_difference`` used as the coincidence time window
    _COINCIDENCE_TOLERANCE = 0.25

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        path_diff: int,
        phase: float = 0.0,
        phase_error: float = 0.0,
    ) -> None:
        """Construct a MichelsonInterferometer.

        Args:
            name (str): instance label.
            timeline (Timeline): simulation timeline.
            path_diff (int): path length difference in ps.  Set to
                ``round(1e12 / f_clock)`` for the COW monitoring line.
            phase (float): interferometer phase in radians (default 0).
                Changing this via :meth:`set_phase` simulates temperature
                tuning as in the experiment.
            phase_error (float): standard deviation (rad) of additive
                Gaussian phase noise applied to each two-photon event
                (default 0).
        """
        Entity.__init__(self, name, timeline)
        self.path_difference: int = path_diff
        self.phase: float = phase
        self.phase_error: float = phase_error

        self._photon_buffer: Optional["Photon"] = None
        self._buffer_time: int = -1

    # ------------------------------------------------------------------
    # Entity interface
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialise (or re-initialise) the interferometer state.

        Called by :meth:`Timeline.init` before simulation starts.
        """
        assert len(self._receivers) == 2, (
            f"MichelsonInterferometer '{self.name}' must be connected to "
            "exactly 2 receivers: _receivers[0]=DM1 (constructive) and "
            "_receivers[1]=DM2 (destructive)."
        )
        self._photon_buffer = None
        self._buffer_time = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_phase(self, phase: float) -> None:
        """Set the interferometer phase (simulates thermal tuning).

        Args:
            phase (float): new phase in radians.
        """
        self.phase = phase

    def get(self, photon: "Photon", **kwargs) -> None:
        """Receive a photon from the monitoring-line input.

        The photon is conceptually split 50/50 at the input beamsplitter:

        * The *short-arm* copy travels straight through.
        * The *long-arm* copy is delayed by ``path_difference``.

        If a photon is already in the buffer and arrived exactly
        ``path_difference`` ps ago (within the tolerance window), the two
        long-arm / short-arm copies overlap and :meth:`_interfere` is called
        to route the photon to DM1 or DM2 according to the phase.

        If no matching partner exists the incoming photon is stored in the
        buffer and a timeout event is scheduled; on timeout the photon exits
        50/50 (no coherence partner, so no interference fringe).

        Args:
            photon (Photon): incoming photon (must be a non-null photon from
                the monitoring beamsplitter).
        """
        now = self.timeline.now()
        tol = int(self.path_difference * self._COINCIDENCE_TOLERANCE)

        # Check whether this photon is the short-arm partner of the buffered
        # long-arm photon (i.e. they should overlap at the recombination BS).
        if self._photon_buffer is not None:
            dt = now - self._buffer_time
            if abs(dt - self.path_difference) <= tol:
                # --- Two-photon interference event ---
                stored = self._photon_buffer
                self._photon_buffer = None
                self._buffer_time = -1
                self._interfere(photon, stored)
                return

        # No matching partner yet: place this photon in the long-arm buffer
        # and schedule a timeout for when the delay expires.
        self._photon_buffer = photon
        self._buffer_time = now
        timeout = now + self.path_difference + tol + 1

        process = Process(self, "_timeout", [photon, now])
        event = Event(timeout, process)
        self.timeline.schedule(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _timeout(self, photon: "Photon", scheduled_at: int) -> None:
        """Handle expiration of the long-arm buffer — no interference partner.

        If the photon is still in the buffer (i.e. it was not consumed by an
        interference event), route it 50/50 to DM1 or DM2.

        Args:
            photon (Photon): the photon that was buffered.
            scheduled_at (int): timeline time (ps) when the photon was
                buffered — used to confirm the buffer has not been replaced.
        """
        if self._photon_buffer is not photon or self._buffer_time != scheduled_at:
            # Already consumed by an interference event, or a new photon
            # has since arrived — nothing to do.
            return

        self._photon_buffer = None
        self._buffer_time = -1
        
        # Isolated photon: random 50/50 exit (contributes equally to both
        # detectors and therefore does not bias the visibility estimate).
        idx = int(self.get_generator().random() > 0.5)
        self._receivers[idx].get(photon)


    def _interfere(self, photon: "Photon", stored: "Photon") -> None:
        """Apply two-photon interference and route the photon.

        For two coherent pulses with relative phase φ the interference
        probabilities at the two output ports are:

            P(DM1) = cos²(φ / 2)   — constructive
            P(DM2) = sin²(φ / 2)   — destructive

        These sum to 1 for any φ.  With φ = 0 (perfect laser coherence and
        stable interferometer) all photons exit through DM1 → V = 1.

        Optional Gaussian phase noise (``phase_error`` > 0) models imperfect
        thermal stabilisation or laser linewidth.

        Args:
            photon (Photon): the arriving short-arm photon that triggers the
                interference event.
            stored (Photon): buffered long-arm photon paired with `photon`.
        """
        is_coherent = (
            getattr(photon, "coherent", True)
            and getattr(stored, "coherent", True)
        )
        if not is_coherent:
            # Fase relativa aleatória → roteamento 50/50
            idx = int(self.get_generator().random() > 0.5)
            self._receivers[idx].get(photon)
            return
        
        phi = self.phase
        if self.phase_error > 0.0:
            phi += float(self.get_generator().normal(0.0, self.phase_error))
        prob_dm1 = cos(phi / 2.0) ** 2   # = (1 + cos φ) / 2
        
        # prob_dm2 = 1 - prob_dm1          # = (1 − cos φ) / 2

        if self.get_generator().random() < prob_dm1:
            self._receivers[0].get(photon)   # DM1 — constructive
        else:
            self._receivers[1].get(photon)   # DM2 — destructive

    # ------------------------------------------------------------------
    # Visibility helper (class-level, used by COW protocol)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_visibility(
        n_dm1: int,
        n_dm2: int,
    ) -> float:
        """Compute the Michelson fringe visibility from detector counts.

        Implements Eq. (3) of Stucki et al. (2005):

            V = (p_DM1 − p_DM2) / (p_DM1 + p_DM2)

        where p_DMj is the *probability* that detector DMj fired at a time
        where only DM1 should have fired.  In practice this is estimated
        from the ratio of detection counts accumulated over many decoy
        sequences.

        Args:
            n_dm1 (int): number of detection events at DM1 during the
                monitoring window.
            n_dm2 (int): number of detection events at DM2 during the
                monitoring window (ideally 0 for perfect coherence).

        Returns:
            float: visibility V ∈ [−1, 1].  Returns 1.0 when both counts
            are zero (no photons detected — undefined case, assume ideal).
        """
        total = n_dm1 + n_dm2
        if total == 0:
            return 1.0
        return (n_dm1 - n_dm2) / total

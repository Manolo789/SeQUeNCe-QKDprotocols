"""QSDetector for the COW QKD protocol.

Bob's detector assembly in the COW protocol (Stucki et al., 2005) consists
of two optical paths fed by a **non-equilibrated beamsplitter** with
transmission coefficient t_B ≈ 1:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   incoming       ┌─────────┐  t_B   ─── DB (dataline)       │
│   photon  ──────►│   BS    │                                │
│                  │ (t_B,   │  1-t_B ─── Michelson ─► DM1    │
│                  │ 1-t_B)  │             interferometer     │
│                  └─────────┘          └─────────────► DM2   │
└─────────────────────────────────────────────────────────────┘

* **Dataline** (DB, ``detectors[0]``): receives transmitted photons with
  probability t_B; used to establish the raw key by arrival-time measurement.
* **Monitoring line**: receives reflected photons with probability 1 − t_B;
  fed into the :class:`~sequence.components.michelson_interferometer.MichelsonInterferometer`
  whose two output detectors are DM1 (``detectors[1]``) and DM2
  (``detectors[2]``).

Visibility measurement
----------------------
After each key-generation session the protocol calls
:meth:`get_session_visibility` which returns the fringe visibility
estimated from the DM1 and DM2 detection counts accumulated **since the
last session reset** — matching the definition in Eq. (3) of Stucki et al.

The dataline detection times are retrieved via the standard
:meth:`get_photon_times` interface (index 0 → DB only), maintaining
compatibility with :meth:`QKDNode.get_bits`.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from .photon import Photon

from .detector import Detector, QSDetector
from .michelson_interferometer import MichelsonInterferometer


class QSDetectorCOW(QSDetector):
    """Bob's COW detector assembly.

    Three single-photon detectors are managed:
        * ``detectors[0]`` — DB:  dataline direct detector
        * ``detectors[1]`` — DM1: monitoring line, constructive port
        * ``detectors[2]`` — DM2: monitoring line, destructive port

    The Michelson interferometer is stored as ``self.interferometer``.

    Attributes:
        name (str): component label.
        timeline (Timeline): simulation timeline.
        t_B (float): beamsplitter transmission to the dataline (0 < t_B ≤ 1).
        interferometer (MichelsonInterferometer): monitoring-line interferometer.
        detectors (list[Detector]): [DB, DM1, DM2].
        trigger_times (list[list[int]]): detection times per detector.
        _session_dm1 (int): DM1 interference events accumulated over the
            entire key-generation session (reset only by
            :meth:`reset_session_counters`).
        _session_dm2 (int): DM2 interference events accumulated over the
            entire key-generation session.

    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        path_diff: int,
        t_B: float = 0.9,
        interferometer_phase: float = 0.0,
        interferometer_phase_error: float = 0.0,
    ) -> None:
        """Construct a QSDetectorCOW.

        Args:
            name (str): component label.
            timeline (Timeline): simulation timeline.
            path_diff (int): Michelson path difference in ps — must equal the
                light-source slot period (``round(1e12 / f_clock)``).
            t_B (float): beamsplitter transmission to the dataline detector
                (default 0.9, so 10 % of photons go to monitoring).
            interferometer_phase (float): initial Michelson phase in radians
                (default 0 — constructive at DM1).
            interferometer_phase_error (float): std-dev of per-event phase
                noise in the Michelson (default 0).
        """
        QSDetector.__init__(self, name, timeline)

        if not (0.0 < t_B <= 1.0):
            raise ValueError(f"t_B must be in (0, 1], got {t_B}")
        self.t_B = t_B

        # --- detectors ---
        self.detectors = [
            Detector(f"{name}.DB",  timeline),   # [0] dataline
            Detector(f"{name}.DM1", timeline),   # [1] monitoring constructive
            Detector(f"{name}.DM2", timeline),   # [2] monitoring destructive
        ]

        # --- Michelson interferometer ---
        self.interferometer = MichelsonInterferometer(
            f"{name}.Michelson",
            timeline,
            path_diff=path_diff,
            phase=interferometer_phase,
            phase_error=interferometer_phase_error,
        )
        self.interferometer.add_receiver(self.detectors[1])   # DM1
        self.interferometer.add_receiver(self.detectors[2])   # DM2

        # Wire detectors back to self so trigger_times are populated
        for d in self.detectors:
            d.attach(self)

        self.trigger_times: list[list[int]] = [[], [], []]
        self._dm1_count: int = 0
        self._dm2_count: int = 0


        
        # Session-level counters — accumulated across all bursts within
        # one key-generation run.  Reset only by reset_session_counters().
        # This implements the paper's Eq. (3): V is estimated from counts
        # accumulated over the entire communication session, not per burst.
        self._session_dm1: int = 0
        self._session_dm2: int = 0



        # Register all components for init()
        self.components = self.detectors + [self.interferometer]

    # ------------------------------------------------------------------
    # QSDetector interface
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialise all sub-components."""
        for component in self.components:
            component.owner = self.owner
        self.interferometer.owner = self   # ensure record_interference is reachable
        self.interferometer.init()
        for d in self.detectors:
            d.init()
        self.trigger_times = [[], [], []]
        self._dm1_count = 0
        self._dm2_count = 0

        
        self._session_dm1 = 0
        self._session_dm2 = 0


    def get(self, photon: "Photon", **kwargs) -> None:
        """Route an incoming photon to the dataline or monitoring line.

        The non-equilibrated beamsplitter is modelled probabilistically:
        * with probability ``t_B``   → DB  (dataline)
        * with probability ``1-t_B`` → Michelson interferometer (monitoring)

        Args:
            photon (Photon): photon arriving at Bob's input port.
        """
        rng = self.get_generator()
        if rng.random() < self.t_B:
            # Transmitted → dataline detector
            self.detectors[0].get(photon)
        else:
            # Reflected → monitoring line interferometer
            self.interferometer.get(photon)

    def trigger(self, detector: "Detector", info: dict[str, Any]) -> None:
        """Record a detection event from one of the three detectors.

        Also maintains the DM1/DM2 counters used by
        :meth:`get_monitoring_visibility`.

        Args:
            detector (Detector): the detector that fired.
            info (dict): message from the detector; must contain ``'time'``.
        """
        idx = self.detectors.index(detector)
        self.trigger_times[idx].append(info["time"])

    def record_interference(self, port: int) -> None:
        """Record a genuine two-photon interference event.
 
        Called by :class:`MichelsonInterferometer._interfere` immediately
        before routing the photon to ``_receivers[port]``. This is the
        only place where counters are incremented, ensuring that isolated
        photons exiting via the timeout path are excluded from the
        visibility calculation — matching the paper's definition of
        p(D_Mj) as "probability that D_Mj fired at a time where only
        DM1 should have fired" (Eq. 3).

        Both per-burst and session-level counters are incremented.
        
        Args:
            port (int): receiver index — 0 for DM1 (constructive),
                1 for DM2 (destructive).
        """

        if port == 0:
            self._dm1_count += 1
            self._session_dm1 += 1
        elif port == 1:
            self._dm2_count += 1
            self._session_dm2 += 1



    # ------------------------------------------------------------------
    # Dataline detection retrieval (compatible with QKDNode.get_bits)
    # ------------------------------------------------------------------

    def get_photon_times(self) -> list[list[int]]:
        """Return and reset detection time lists for all detectors.

        Returns:
            list[list[int]]: ``[db_times, dm1_times, dm2_times]``.
        """
        times = self.trigger_times
        self.trigger_times = [[], [], []]
        return times

    # ------------------------------------------------------------------
    # Monitoring line visibility
    # ------------------------------------------------------------------
    def get_monitoring_visibility(self) -> float:
        v = MichelsonInterferometer.compute_visibility(
            self._dm1_count, self._dm2_count)
        self._dm1_count = 0
        self._dm2_count = 0
        return v
        
    def get_session_visibility(self) -> float:
        """Compute session-accumulated Michelson fringe visibility.

        Implements Eq. (3) of Stucki et al. (2005):

            V = (n_DM1 − n_DM2) / (n_DM1 + n_DM2)

        where counts are accumulated over the **entire key-generation
        session** (all bursts since the last :meth:`reset_session_counters`
        call).  This provides the statistically meaningful estimate that
        the paper assumes.

        Does NOT reset session counters — call :meth:`reset_session_counters`
        explicitly at the beginning of each new key-generation run.

        Returns:
            float: visibility V ∈ [−1, 1], or ``float('nan')`` if no
            interference events were recorded in the session.
        """
        return MichelsonInterferometer.compute_visibility(
            self._session_dm1, self._session_dm2
        )

    def get_session_counts(self) -> tuple:
        """Return the raw session-level DM1 and DM2 counts.

        Useful for debugging and external analysis.

        Returns:
            tuple: (session_dm1, session_dm2).
        """
        return (self._session_dm1, self._session_dm2)

    def reset_session_counters(self) -> None:
        """Reset session-level DM1/DM2 counters.

        Must be called at the beginning of each new key-generation run
        (before the first burst of that run).  This ensures that
        :meth:`get_session_visibility` accumulates counts only within
        one run.
        """
        self._session_dm1 = 0
        self._session_dm2 = 0

    def set_phase(self, phase: float) -> None:
        """Adjust the Michelson interferometer phase (temperature tuning).

        Args:
            phase (float): new phase in radians.
        """
        self.interferometer.set_phase(phase)

    def set_basis_list(
        self, basis_list: list[int], start_time: int, frequency: float
    ) -> None:
        """No-op: COW uses passive direct detection — no basis switching.

        Present to satisfy the :class:`QSDetector` abstract interface.
        """
        pass   # COW requires no active basis choice at Bob's side

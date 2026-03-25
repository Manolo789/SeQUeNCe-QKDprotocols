"""
=======================================================================================
Time-bin encoding model with explicit vacuum state in the SeQUENCe simulator -- License
=======================================================================================

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

=======================================================================================

"""

"""COW-specific time-bin encoding with explicit vacuum state.

This module defines ``time_bin_cow``, a photon-encoding dictionary for the
Coherent One-Way (COW) QKD protocol.  It extends the standard ``time_bin``
encoding used elsewhere in SeQUeNCe with:

1. An explicit **vacuum sentinel** (``VACUUM_STATE``) that ``LightSource``
   (and the patched :class:`COWLightSource`) can recognise as "emit nothing
   this slot".
2. Named *pulse states* (``EARLY_STATE``, ``LATE_STATE``) that correspond to
   the two COW data-bit encodings.
3. A ``DECOY_STATES`` tuple with both pulse states, used for the decoy
   sequence |μ⟩|μ⟩.
4. A convenience ``slot_period`` helper that returns τ in ps from the clock
   frequency.

COW encoding table
------------------
Symbol | Slot 2k−1  | Slot 2k  | Meaning
-------|------------|----------|--------
Bit 0  | |μ⟩        | vacuum   | early-bin photon → bit 0
Bit 1  | vacuum     | |μ⟩      | late-bin photon  → bit 1
Decoy  | |μ⟩        | |μ⟩      | both bins filled → coherence check

The vacuum slot is represented by the sentinel ``VACUUM_STATE = None``.
When the :class:`COWLightSource` (or the patched
:meth:`LightSource.emit`) encounters ``None`` in the state list it advances
the clock without emitting any photon.

Usage example
-------------
::

    from sequence.utils.encoding_cow import time_bin_cow, VACUUM_STATE

    enc = time_bin_cow
    # Build state list for symbols [bit0, bit1, decoy]:
    # bit0  → (early pulse, vacuum)
    # bit1  → (vacuum,      late pulse)
    # decoy → (early pulse, late pulse)
    state_list = [
        enc["early"],  VACUUM_STATE,          # bit 0
        VACUUM_STATE,  enc["late"],            # bit 1
        enc["early"],  enc["late"],            # decoy
    ]
"""

from math import sqrt

import numpy as np
from numpy import array

# ---------------------------------------------------------------------------
# Vacuum sentinel
# ---------------------------------------------------------------------------

#: Sentinel placed in a state list to request zero photon emission for that
#: time slot.  Recognised by :class:`COWLightSource` and
#: :func:`patch_lightsource_for_cow`.
VACUUM_STATE = None


# ---------------------------------------------------------------------------
# Basis vectors (same convention as the existing ``time_bin`` encoding)
# ---------------------------------------------------------------------------

#: Photon in the *early* time bin  — encodes **bit 0** in the COW protocol.
EARLY_STATE: np.ndarray = array([complex(1), complex(0)])

#: Photon in the *late* time bin   — encodes **bit 1** in the COW protocol.
LATE_STATE: np.ndarray = array([complex(0), complex(1)])

#: X-basis superposition states (used internally by the Michelson
#: interferometer to verify phase coherence; not used for data encoding).
X_PLUS_STATE:  np.ndarray = array([complex(1 / sqrt(2)), complex( 1 / sqrt(2))])
X_MINUS_STATE: np.ndarray = array([complex(1 / sqrt(2)), complex(-1 / sqrt(2))])


# ---------------------------------------------------------------------------
# time_bin_cow encoding dictionary
# ---------------------------------------------------------------------------

#: Default slot period in ps (corresponds to 434 MHz from Stucki et al.).
#_DEFAULT_BIN_SEPARATION: int = round(1e12 / 434e6)   # ≈ 2304 ps
_DEFAULT_BIN_SEPARATION = 1400   # ≈ 2304 ps

time_bin_cow: dict = {
    # ---- required by SeQUeNCe infrastructure ----
    "name": "time_bin_cow",

    # Two measurement bases:
    #   bases[0] = Z basis  (direct arrival-time detection)
    #   bases[1] = X basis  (interferometric superposition — monitoring line)
    "bases": [
        [EARLY_STATE, LATE_STATE],
        [X_PLUS_STATE, X_MINUS_STATE],
    ],

    # Separation between the early and late slots within one symbol (ps).
    # This equals one clock period τ and must match the Michelson
    # interferometer's ``path_difference``.
    "bin_separation": _DEFAULT_BIN_SEPARATION,

    # ---- COW-specific convenience fields ----

    #: Pulse state for the early slot  → data bit 0.
    "early": EARLY_STATE,

    #: Pulse state for the late  slot  → data bit 1.
    "late": LATE_STATE,

    #: Vacuum sentinel: no photon emitted for this slot.
    "vacuum": VACUUM_STATE,

    #: State pair for a decoy sequence (|μ⟩ in both slots).
    "decoy": (EARLY_STATE, LATE_STATE),

    # ---- compatibility flags ----
    "keep_photon": False,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def slot_period_ps(clock_frequency_hz: float) -> int:
    """Return the slot period τ in picoseconds for a given clock frequency.

    This value should be used as the ``path_difference`` parameter of the
    :class:`~sequence.components.michelson_interferometer.MichelsonInterferometer`.

    Args:
        clock_frequency_hz (float): light-source clock frequency in Hz
            (e.g. 434e6 for the Stucki et al. experiment).

    Returns:
        int: τ = round(1e12 / f) in ps.
    """
    return round(1e12 / clock_frequency_hz)


def build_cow_state_list(bits: list, is_decoy: list, *, include_vacuum_pulses: bool = True,) -> list:
    """Build the flat state list consumed by :class:`COWLightSource.emit`.

    Each COW *symbol* occupies two consecutive time slots.  The returned
    list has length ``2 * len(bits)`` and alternates (slot_2k-1, slot_2k).

    Args:
        bits (list[int]): data bits (0 or 1) for each symbol.  Ignored for
            decoy positions.
        is_decoy (list[bool]): True at index *i* if symbol *i* is a decoy.
        include_vacuum_pulses (bool): when True (default) vacuum slots are
            represented by ``VACUUM_STATE``; when False a zero-amplitude
            placeholder state is used instead (for light-sources that
            internally handle Poisson statistics).

    Returns:
        list: flat list of states / ``VACUUM_STATE`` sentinels, length
        ``2 * len(bits)``.

    Raises:
        ValueError: if ``bits`` and ``is_decoy`` have different lengths.
    """
    if len(bits) != len(is_decoy):
        raise ValueError(
            f"bits and is_decoy must have the same length "
            f"({len(bits)} vs {len(is_decoy)})"
        )

    vac = VACUUM_STATE if include_vacuum_pulses else EARLY_STATE

    state_list: list = []
    for bit, decoy in zip(bits, is_decoy):
        if decoy:
            # Decoy: both slots occupied → |μ⟩|μ⟩
            state_list.append(EARLY_STATE)
            state_list.append(LATE_STATE)
        elif bit == 0:
            # Bit 0: early-slot pulse, late-slot vacuum → |μ⟩|0⟩
            state_list.append(EARLY_STATE)
            state_list.append(vac)
        else:
            # Bit 1: early-slot vacuum, late-slot pulse → |0⟩|μ⟩
            state_list.append(vac)
            state_list.append(LATE_STATE)

    return state_list

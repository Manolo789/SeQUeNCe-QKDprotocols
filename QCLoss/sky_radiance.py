"""
============================================================================
Implementation of Sky spectral radiance B_sky as a function of
solar elevation + background photons per mode n_B -- License
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

Replaces the binary day/night switch of scenarios.py
(B_sky = 1.5e-6 <-> 1.5e-2 W·m^-2·nm^-1·sr^-1) with a continuous physical
model driven by the solar elevation.

MODEL CHAIN (clear sky):
  1. Solar position (declination, equation of time, hour angle) --
     Spencer's (1971) Fourier series, as in the NOAA solar algorithm;
     see also Meeus, "Astronomical Algorithms" (1998).
  2. Relative air mass -- Kasten & Young (1989), valid down to the horizon.
  3. Spectral diffuse irradiance on the horizontal plane -- Rayleigh and
     aerosol components of the SPCTRAL2 model of Bird & Riordan (1986),
     evaluated at the link wavelength. In the atmospheric windows used in
     QKD (770-860 nm, 1550 nm) the O3, H2O and uniform-gas transmittances
     are ~1 and are omitted (documented assumption; for a lambda inside an
     absorption band use full SPCTRAL2 or MODTRAN).
     Aerosol turbidity from the Angstrom law tau_a = beta*lam_um^(-alpha)
     (Angstrom 1964; typical beta in Iqbal, "An Introduction to Solar
     Radiation", 1983: 0.05 clean / 0.1 average / 0.2 turbid).
  4. Sky radiance: isotropic-sky assumption L = E_dif/pi
     (Liou, "An Introduction to Atmospheric Radiation", 2002, para. 1.1;
     a first-order approximation -- a real clear sky varies with the
     scattering angle to the Sun, typically by a factor of 2-5).
  5. Twilight (-18 deg < h < 0 deg): log-linear decay between L(h=0) and
     the night floor at h = -18 deg, consistent with the ~3-4 decades of
     zenith-radiance drop measured by Rozenberg, "Twilight: A Study in
     Atmospheric Optics" (1966); see also Patat, A&A 400, 1183 (2003).
  6. Night (h <= -18 deg): configurable floor. Reference values
     (Er-long Miao et al., New J. Phys. 7, 215 (2005); Bourgoin et al.,
     New J. Phys. 15, 023006 (2013); Pirandola, PRR 3, 023130 (2021)):
       full moon, clear sky : ~1.5e-3 W·m^-2·um^-1·sr^-1 = 1.5e3 SI
       new moon,  clear sky : ~1.5e-6 W·m^-2·um^-1·sr^-1 = 1.5e0 SI

UNITS: strict SI. Spectral radiance in W·m^-2·m^-1·sr^-1 (per METRE of
wavelength). Conversions:
  1 W·m^-2·nm^-1·sr^-1 = 1e9 SI  |  1 W·m^-2·um^-1·sr^-1 = 1e6 SI
The former fork value B_sky = 1.5e-2 (per nm) equals 1.5e7 SI.
"""

import math
import warnings
from datetime import datetime, timezone

H_PLANCK = 6.62607015e-34   # J·s
C_LIGHT = 2.99792458e8      # m/s

# ---------------------------------------------------------------------------
# Extraterrestrial solar irradiance E0(lambda) -- anchors of the ASTM E490
# spectrum (AM0), in W·m^-2·m^-1 (SI). Linearly interpolated in between.
# ---------------------------------------------------------------------------
_E0_AM0 = {  # λ [m] : E0 [W·m⁻²·m⁻¹]
    400e-9: 1.60e9, 500e-9: 1.95e9, 550e-9: 1.86e9, 650e-9: 1.55e9,
    780e-9: 1.17e9, 850e-9: 0.97e9, 1000e-9: 0.75e9, 1064e-9: 0.65e9,
    1310e-9: 0.37e9, 1550e-9: 0.265e9,
}


def extraterrestrial_irradiance(wavelength: float) -> float:
    """Extraterrestrial spectral irradiance E0 at the top of the atmosphere.

    Args:
        wavelength: wavelength [m].

    Returns:
        float: E0 [W·m^-2·m^-1], linearly interpolated between the ASTM
        E490 anchors and clamped outside their range.
    """
    lams = sorted(_E0_AM0)
    if wavelength <= lams[0]:
        return _E0_AM0[lams[0]]
    if wavelength >= lams[-1]:
        return _E0_AM0[lams[-1]]
    for lo, hi in zip(lams, lams[1:]):
        if lo <= wavelength <= hi:
            t = (wavelength - lo) / (hi - lo)
            return _E0_AM0[lo] + t * (_E0_AM0[hi] - _E0_AM0[lo])


# ---------------------------------------------------------------------------
# 1. Solar position -- Spencer (1971) / NOAA
# ---------------------------------------------------------------------------
def solar_elevation(when_utc: datetime, latitude: float,
                    longitude: float) -> float:
    """Solar elevation h [rad], negative below the horizon.

    Args:
        when_utc: datetime WITH UTC tzinfo (naive values are read as UTC).
        latitude: site latitude [rad], positive North.
        longitude: site longitude [rad], positive East.

    Returns:
        float: solar elevation [rad].

    References: Spencer, Search 2, 172 (1971); NOAA Solar Calculator;
    Meeus (1998). Accuracy ~0.01 rad, sufficient for radiometry.
    """
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    when_utc = when_utc.astimezone(timezone.utc)
    doy = when_utc.timetuple().tm_yday
    frac_h = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    # year angle [rad]
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (frac_h - 12.0) / 24.0)
    # solar declination [rad] -- Spencer (1971)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    # equation of time [min] -- Spencer (1971)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g)
                    - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    # true solar time and hour angle
    tst = frac_h + eqt / 60.0 + math.degrees(longitude) / 15.0
    hour_angle = math.radians(15.0 * (tst - 12.0))
    sin_h = (math.sin(latitude) * math.sin(decl)
             + math.cos(latitude) * math.cos(decl) * math.cos(hour_angle))
    return math.asin(max(-1.0, min(1.0, sin_h)))


def kasten_young_airmass(elevation: float) -> float:
    """Massa de ar relativa M(h) — Kasten & Young (1989).

    M = 1/[sin h + 0,50572·(h° + 6,07995)^(−1,6364)], finita no horizonte.
    """
    h_deg = math.degrees(elevation)
    return 1.0 / (math.sin(elevation)
                  + 0.50572 * (h_deg + 6.07995) ** (-1.6364))


# ---------------------------------------------------------------------------
# 2. Clear-sky diffuse radiance -- simplified SPCTRAL2 (Bird & Riordan)
# ---------------------------------------------------------------------------
def _rayleigh_od(wavelength: float, pressure: float) -> float:
    """Vertical Rayleigh optical depth tau_R, Bird & Riordan (1986).

    tau_R = (P/P0)/[lam^4*(115.6406 - 1.335/lam^2)], lam in um.

    Args:
        wavelength: wavelength [m].
        pressure: station pressure [Pa].

    Returns:
        float: dimensionless vertical Rayleigh optical depth.
    """
    lam_um = wavelength * 1e6                    # m -> um (published form)
    return (pressure / 101325.0) / (lam_um ** 4
                                    * (115.6406 - 1.335 / lam_um ** 2))


def clear_sky_radiance(wavelength: float, elevation: float,
                       pressure: float = 101325.0,
                       angstrom_beta: float = 0.10,
                       angstrom_alpha: float = 1.3,
                       ssa: float = 0.90, asym_g: float = 0.65) -> float:
    """Clear-sky spectral radiance L [W·m^-2·m^-1·sr^-1] for h > 0.

    Rayleigh (I_r) and aerosol (I_a) diffuse components of SPCTRAL2
    [Bird & Riordan 1986, Eqs. 3-9 to 3-13], with T_O3 = T_H2O = T_gas = 1
    (QKD atmospheric windows) and no ground-reflection term;
    L = (I_r + I_a)/pi (isotropic sky, Liou 2002).

    Args:
        wavelength: wavelength [m].
        elevation: solar elevation [rad]; returns 0 at or below the horizon.
        pressure: station pressure [Pa].
        angstrom_beta: Angstrom turbidity coefficient beta.
        angstrom_alpha: Angstrom exponent alpha, in tau_a = beta*lam_um^-alpha.
        ssa: aerosol single-scattering albedo (continental default).
        asym_g: aerosol asymmetry parameter g (continental default).

    Returns:
        float: sky radiance [W·m^-2·m^-1·sr^-1].
    """
    if elevation <= 0:
        return 0.0
    cosZ = math.sin(elevation)
    M = kasten_young_airmass(elevation)
    E0 = extraterrestrial_irradiance(wavelength)

    tau_r = _rayleigh_od(wavelength, pressure)
    lam_um = wavelength * 1e6
    tau_a = angstrom_beta * lam_um ** (-angstrom_alpha)

    T_r = math.exp(-tau_r * M)
    T_aa = math.exp(-(1.0 - ssa) * tau_a * M)     # aerosol absorption
    T_as = math.exp(-ssa * tau_a * M)             # aerosol scattering

    # Downward-scattered aerosol fraction F_s [Bird & Riordan 1986]
    alg = math.log(1.0 - asym_g)
    afs = alg * (1.459 + alg * (0.1595 + 0.4129 * alg))
    bfs = alg * (0.0783 + alg * (-0.3824 - 0.5874 * alg))
    F_s = 1.0 - 0.5 * math.exp((afs + bfs * cosZ) * cosZ)

    I_r = E0 * cosZ * T_aa * 0.5 * (1.0 - T_r ** 0.95)
    I_a = E0 * cosZ * T_aa * (T_r ** 1.5) * (1.0 - T_as) * F_s
    return (I_r + I_a) / math.pi


# ---------------------------------------------------------------------------
# 3. Continuous B_sky: day + twilight + night
# ---------------------------------------------------------------------------
B_NIGHT_NEW_MOON = 1.5e0    # W·m^-2·m^-1·sr^-1 (= 1.5e-6 um^-1) [Miao05/Bourgoin13]
B_NIGHT_FULL_MOON = 1.5e3   # W·m^-2·m^-1·sr^-1 (= 1.5e-3 um^-1) [Pirandola21]
_TWILIGHT_END = math.radians(-18.0)   # astronomical twilight


def b_sky(wavelength: float, elevation: float,
          pressure: float = 101325.0,
          angstrom_beta: float = 0.10,
          b_night: float = B_NIGHT_FULL_MOON,
          cloud_factor: float = 1.0) -> float:
    """Sky spectral radiance B_sky [W·m^-2·m^-1·sr^-1] vs solar elevation.

    Regimes:
      h > 0            : SPCTRAL2 clear sky (clear_sky_radiance) times
                         cloud_factor (an overcast sky raises the diffuse
                         radiance by ~10x; Bourgoin et al. 2013, Tab. 2)
      -18 deg < h <= 0 : log-linear interpolation between L(h->0+) and
                         b_night [Rozenberg 1966: ~3-4 decades over 18 deg]
      h <= -18 deg     : b_night (new/full moon, see B_NIGHT_*)

    Args:
        wavelength: wavelength [m].
        elevation: solar elevation [rad].
        pressure: station pressure [Pa].
        angstrom_beta: Angstrom turbidity coefficient.
        b_night: night radiance floor [SI].
        cloud_factor: multiplier applied to the clear-sky daytime radiance.

    Returns:
        float: B_sky [W·m^-2·m^-1·sr^-1].
    """
    L_day_horizon = clear_sky_radiance(
        wavelength, math.radians(0.5), pressure, angstrom_beta) * cloud_factor
    if elevation > 0:
        L = clear_sky_radiance(wavelength, elevation, pressure,
                               angstrom_beta) * cloud_factor
        return max(L, b_night)
    if elevation > _TWILIGHT_END:
        f = elevation / _TWILIGHT_END            # 0 at sunset -> 1 at the end
        logL = ((1.0 - f) * math.log10(max(L_day_horizon, b_night))
                + f * math.log10(b_night))
        return 10.0 ** logL
    return b_night


def b_sky_at(when_utc: datetime, latitude: float, longitude: float,
             wavelength: float, **kwargs) -> float:
    """Convenience wrapper: B_sky from a UTC instant and site coordinates.

    Intended for direct use in the ``diurnal_profile()`` of scenarios.py.

    Args:
        when_utc: instant in UTC.
        latitude: site latitude [rad].
        longitude: site longitude [rad].
        wavelength: wavelength [m].
        **kwargs: forwarded to :func:`b_sky` (pressure, b_night, ...).

    Returns:
        float: B_sky [W·m^-2·m^-1·sr^-1].
    """
    h = solar_elevation(when_utc, latitude, longitude)
    return b_sky(wavelength, h, **kwargs)


# ---------------------------------------------------------------------------
# 4. Background photons per detection mode
# ---------------------------------------------------------------------------
def n_background(wavelength: float, filter_bandwidth: float,
                 detection_gate: float, fov_solid_angle: float,
                 receiver_radius: float, B_sky_si: float) -> float:
    """n_B -- dimensionless background photons per detection mode.

    Pirandola, PRR 3, 023130 (2021), Eq. (32):
        n_B = pi*lambda*Gamma_R*B_sky/(h*c),
        Gamma_R = d_lambda * d_t * Omega_fov * a_R^2

    Args (ALL IN SI):
        wavelength       [m]
        filter_bandwidth [m]   spectral filter width (1 nm -> 1e-9)
        detection_gate   [s]   temporal ACCEPTANCE WINDOW of a click, i.e.
                               the gate matched to the pulse or to the
                               detector resolution -- NOT the dead time
                               1/count_rate. With the parameters of this
                               fork (time_resolution = 1 ns,
                               1/count_rate = 50 ns) using the dead time
                               would overestimate the background by 50x;
                               see :func:`detection_gate_from_detector`.
        fov_solid_angle  [sr]  receiver field-of-view solid angle
        receiver_radius  [m]   a_R
        B_sky_si         [W·m^-2·m^-1·sr^-1] output of b_sky / b_sky_at

    Returns:
        float: mean number of background photons per detection mode.
    """
    gamma_R = (filter_bandwidth * detection_gate * fov_solid_angle
               * receiver_radius ** 2)
    return math.pi * wavelength * gamma_R * B_sky_si / (H_PLANCK * C_LIGHT)


def detection_gate_from_detector(time_resolution_ps: float,
                                 pulse_width_s: float = 0.0) -> float:
    """Recommended detection gate [s] for :func:`n_background`.

    The gate is matched to the detector temporal resolution (SeQUeNCe's
    ``time_resolution`` parameter, in ps) or to the pulse width, whichever
    is larger.

    Args:
        time_resolution_ps: detector time resolution [ps].
        pulse_width_s: optical pulse width [s].

    Returns:
        float: acceptance window [s].
    """
    return max(time_resolution_ps * 1e-12, pulse_width_s)

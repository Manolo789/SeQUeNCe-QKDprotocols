"""
============================================================================
Implementation of quantum channel loss calculation -- License
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

UNIT CONVENTION (strict SI on every public interface):
    length ............... m        temperature ........ K
    time ................. s        pressure ........... Pa
    speed ................ m/s      dynamic viscosity .. Pa·s
    wavelength ........... m        precipitation rate . m/s
    Cn² .................. m^(-2/3) spectral radiance .. W·m⁻²·m⁻¹·sr⁻¹
Conversions required by empirical formulas published in other units (e.g.
visibility in km in Kim's model, R in mm/h in Marshall-Palmer) are done
INTERNALLY, with an explicit comment at the conversion point.

MODELLING CHOICES (and why the naive formulations are not used):
  -- Rain attenuation: Stokes' law (Re<<1) is invalid for raindrops
       (Re~10^2-10^3; it gives v_t~118 m/s for D=2 mm against ~6.5 m/s
       measured). The model uses the Marshall-Palmer (1948) drop-size
       distribution with Q_ext=2 (extinction paradox, x=piD/lambda>~10^3)
       in closed form, the empirical terminal velocity of Atlas et al.
       (1973), and cross-checks against Carbonneau's empirical form.
  -- Cn²: horizontal links near the ground take the SURFACE value scaled
       by h^(-4/3) (unstable daytime) or h^(-2/3) (night-time), with h
       ABOVE GROUND LEVEL. Feeding that surface value into the exponential
       term of the Hufnagel-Valley profile would inflate it by ~37x, so
       the HV profile lives in a separate function for slant paths.
  -- inner_scale: u* is unambiguously in m/s, and the weak<->strong regime
       criterion uses the plane-wave coherence length with the 1.46 factor:
       rho0 = (1.46*Cn²*k²*L)^(-3/5).
  -- polarization_fidelity() is deliberately absent: the polarisation
       fidelity is to be taken as 1.0 in the simulator (turbulence
       preserves single-mode polarisation; see Fedrizzi et al., Nat. Phys.
       5, 389 (2009), a 144 km link with <1% error). A complete
       formulation would first require studying how aerosols degrade the
       polarisation state.

MAIN REFERENCES:
  [AP05]  Andrews & Phillips, "Laser Beam Propagation Through Random
          Media", 2nd ed., SPIE (2005).
  [SK92]  Sadot & Kopeika, Opt. Eng. 31, 200 (1992) — macroscale model
          of Cn² (T in Kelvin).
  [Ben04] Bendersky, Kopeika & Blaunstein, Appl. Opt. 43, 4070 (2004).
  [Wyn71] Wyngaard, Izumi & Collins, J. Opt. Soc. Am. 61, 1646 (1971) —
          Cn² ∝ h^(-4/3) (unstable) and h^(-2/3) (neutral/stable).
  [MP48]  Marshall & Palmer, J. Meteor. 5, 165 (1948) — N(D)=N0·e^(-ΛD).
  [Atl73] Atlas, Srivastava & Sekhon, Rev. Geophys. 11, 1 (1973) —
          v(D)=9.65−10.3·e^(−600·D) m/s (D in m; 0.6–5.8 mm).
  [GK49]  Gunn & Kinzer, J. Meteor. 6, 243 (1949) — measured v_t.
  [Carb98] Carbonneau & Wisely, Proc. SPIE 3232 (1998) —
          A[dB/km]=1.076·R^0.67 (R in mm/h).
  [HQ73]  Hale & Querry, Appl. Opt. 12, 555 (1973) — complex refractive
          index of water, 0.2–200 µm (replaces Quan & Fry, valid only for
          sea water in the visible and ignoring the imaginary part).
  [Kim01] Kim, McArthur & Korevaar, Proc. SPIE 4214 (2001) —
          visibility-based attenuation.
  [HV]    Hufnagel–Valley 5/7, see [AP05] ch. 12.
  [GP22]  Ghalaii & Pirandola, Commun. Phys. 5, 38 (2022).
"""

from typing import Optional
import math
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8        # m/s
H_PLANCK = 6.62607015e-34     # J·s
R_GAS = 8.314462618           # J/(mol·K)
M_AIR = 0.0289645             # kg/mol
KAPPA_VK = 0.4                # von Karman constant
G_STD = 9.80665               # m/s²


# ===========================================================================
# 1. Micrometeorology (SI)
# ===========================================================================
def f_velocity(wind_speed: float, T_classification: int, height_ag: float) -> float:
    """Friction velocity u* [m/s] from the neutral logarithmic profile.

    u* = kappa*U(z) / ln(z/z0)  [Holton 2004; WMO Guide 2024, z0 table]

    Args:
        wind_speed: wind speed at height_ag [m/s]
        T_classification: WMO terrain class (1-8)
        height_ag: measurement height ABOVE GROUND LEVEL [m]

    Returns:
        float: u* [m/s], strictly SI (no cm/s factor of 100).

    Raises:
        ValueError: if height_ag is not above the terrain roughness z0.
    """
    z0 = {1: 0.0002, 2: 0.005, 3: 0.03, 4: 0.10,
          5: 0.25, 6: 0.5, 7: 1.0, 8: 2.0}[T_classification]
    if height_ag <= z0:
        raise ValueError("height_ag must be larger than the terrain z0.")
    return KAPPA_VK * wind_speed / math.log(height_ag / z0)


def viscosity_sutherland(temperature: float) -> float:
    """Dynamic viscosity of air [Pa·s] from Sutherland's equation (1893).

    mu = mu0*((T0+S)/(T+S))*(T/T0)^(3/2), mu0=1.716e-5 Pa·s, T0=273.15 K,
    S=110.4 K. The result is SI, not the CGS poise of the usual tables.

    Args:
        temperature: air temperature [K].

    Returns:
        float: dynamic viscosity [Pa·s].
    """
    return 1.716e-5 * ((273.15 + 110.4) / (temperature + 110.4)) \
        * (temperature / 273.15) ** 1.5


def air_density(pressure: float, temperature: float) -> float:
    """Air density [kg/m³] from the ideal gas law.

    Args:
        pressure: air pressure [Pa].
        temperature: air temperature [K].

    Returns:
        float: density [kg/m³].
    """
    return pressure * M_AIR / (R_GAS * temperature)


def outer_scale(height_ag: float) -> float:
    """Outer turbulence scale L0 [m], from [AP05] p.483 / Lukin (2005).

    Args:
        height_ag: link height ABOVE GROUND LEVEL [m].

    Returns:
        float: outer scale L0 [m].
    """
    h = height_ag
    if h <= 1:
        return 0.4
    if h <= 25:
        return 0.4 * h
    if h <= 1000:
        return 2.0 * math.sqrt(h)
    return 2.0 * math.sqrt(1000.0)


def inner_scale(temperature: float, pressure: float, friction_velocity: float,
                height_ag: float, viscosity: Optional[float] = None) -> float:
    """Inner turbulence scale l0 [m] (Tatarskii), in SI.  [AP05] p.57-82.

    l0 = 7.4*(nu^3/eps)^(1/4); nu = mu/rho; eps = u*^3/(kappa*h) in the
    surface layer. ``friction_velocity`` is unambiguously in m/s: mixing it
    with cm/s changes l0 by a factor of ~30 (119 mm vs 3.8 mm).

    Args:
        temperature: air temperature [K].
        pressure: air pressure [Pa].
        friction_velocity: u* [m/s].
        height_ag: link height ABOVE GROUND LEVEL [m].
        viscosity: dynamic viscosity [Pa·s]; derived from T when None.

    Returns:
        float: inner scale l0 [m].
    """
    if viscosity is None:
        viscosity = viscosity_sutherland(temperature)
    nu = viscosity / air_density(pressure, temperature)          # m²/s
    eps = friction_velocity ** 3 / (KAPPA_VK * height_ag)        # m²/s³
    return 7.4 * (nu ** 3 / eps) ** 0.25


def wind_speed_perp(height_asl: float, ground_wind: float,
                    slew_rate: float = 0.0) -> float:
    """Transverse wind V(h) [m/s], Bufton model [AP05] ch.12 Eq.(3).

    Args:
        height_asl: altitude ABOVE SEA LEVEL [m]; the ~9.4 km tropospheric
            jet term of the model is referred to sea level.
        ground_wind: wind speed at ground level [m/s].
        slew_rate: angular slew rate of the terminal [rad/s], 0 for a
            fixed ground-to-ground link.

    Returns:
        float: transverse wind speed [m/s].
    """
    tropo_jet = 30.0 * math.exp(-((height_asl - 9400.0) / 4800.0) ** 2)
    return slew_rate * height_asl + ground_wind + tropo_jet


# ===========================================================================
# 2. Cn²
# ===========================================================================
def _temporal_hour_weight(hour: float, sunrise: float, sunset: float) -> float:
    """Temporal-hour weight W of Sadot-Kopeika/Bendersky [SK92, Ben04].

    Args:
        hour: local time in fractional hours.
        sunrise, sunset: local solar times in fractional hours.

    Returns:
        float: the tabulated weight W entering the Cn² regression.
    """
    th = 12.0 * (hour - sunrise) / (sunset - sunrise)
    table = [(-4, 0.11), (-3, 0.11), (-2, 0.07), (-1, 0.08), (0, 0.06),
             (1, 0.05), (2, 0.10), (3, 0.51), (4, 0.75), (5, 0.95),
             (6, 1.00), (7, 0.90), (8, 0.80), (9, 0.59), (10, 0.32),
             (11, 0.22), (12, 0.10), (13, 0.08)]
    for upper, w in table:
        if th <= upper:
            return w
    return 0.13


def cn2_surface_sadot_kopeika(hour: float, sunrise: float, sunset: float,
                              temperature: float, wind_speed: float,
                              relative_humidity: float) -> float:
    """Cn² [m^(-2/3)] at the REFERENCE HEIGHT h_ref = 15 m above ground.

    Macroscale regression of [SK92]/[Ben04]:
        Cn² = 3.8e-14*W + 2e-15*T - 2.5e-15*U + 1.2e-15*U^2 - 8.5e-17*U^3
              - 2.8e-15*RH + 2.9e-17*RH^2 - 1.1e-19*RH^3 - 5.3e-13

    ATTENTION: this is a FIRST APPROXIMATION. The regression is only valid
    for 282 K <= T <= 308 K, 0 <= U <= 10 m/s and 14% <= RH <= 92%
    [Ben04]; outside that domain the result is an extrapolation and a
    RuntimeWarning is issued. A model tuned to the simulated geographic
    region should eventually replace it.

    Args:
        hour: local time in fractional hours.
        sunrise, sunset: local solar times in fractional hours.
        temperature: air temperature [K].
        wind_speed: wind speed [m/s].
        relative_humidity: relative humidity [%].

    Returns:
        float: Cn² [m^(-2/3)] at 15 m above ground.
    """
    T, U, RH = temperature, wind_speed, relative_humidity
    if not (282.0 <= T <= 308.0 and 0.0 <= U <= 10.0 and 14.0 <= RH <= 92.0):
        warnings.warn(
            f"cn2_surface_sadot_kopeika: (T={T:.1f} K, U={U:.1f} m/s, "
            f"RH={RH:.0f} %) outside the validity domain of the [Ben04] "
            "regression; the result is an extrapolation.", RuntimeWarning)

    W = _temporal_hour_weight(hour, sunrise, sunset)
    cn2 = (3.8e-14 * W
           + 2.0e-15 * T
           - 2.5e-15 * U + 1.2e-15 * U ** 2 - 8.5e-17 * U ** 3
           - 2.8e-15 * RH + 2.9e-17 * RH ** 2 - 1.1e-19 * RH ** 3
           - 5.3e-13)
    if cn2 <= 0:
        warnings.warn("cn2_surface_sadot_kopeika: the regression returned a "
                      "non-physical value (<=0); falling back to 1e-17 "
                      "m^(-2/3). Check the input units/domain.",
                      RuntimeWarning)
        cn2 = 1e-17
    return cn2

CN2_REFERENCE_HEIGHT = 15.0   # m above ground - regression height [Ben04]


def cn2_horizontal_link(height_link_ag: float, hour: float, sunrise: float,
                        sunset: float, temperature: float, wind_speed: float,
                        relative_humidity: float) -> float:
    """Cn² [m^(-2/3)] at the height of a HORIZONTAL LINK, above ground.

    The surface-layer law of [Wyn71] is applied directly:

        Cn²(h) = Cn²(h_ref)*(h/h_ref)^p,  p = -4/3 (daytime, unstable)
                                          p = -2/3 (night, neutral/stable)

    with h = height ABOVE GROUND LEVEL. For the 8 m link of the base
    scenario this gives ~5e-14 at midday, inside the typical near-ground
    daytime range (1e-14 to 1e-13). Injecting the surface value into the
    exponential term of the Hufnagel-Valley profile evaluated at the
    altitude above SEA level would instead annihilate it through
    exp(-h_asl/100) and yield ~8e-16.

    Args:
        height_link_ag: link height ABOVE GROUND LEVEL [m].
        hour: local time in fractional hours.
        sunrise, sunset: local solar times in fractional hours.
        temperature: air temperature [K].
        wind_speed: wind speed [m/s].
        relative_humidity: relative humidity [%].

    Returns:
        float: Cn² [m^(-2/3)] at the link height.
    """
    cn2_ref = cn2_surface_sadot_kopeika(hour, sunrise, sunset, temperature,
                                        wind_speed, relative_humidity)
    is_day = sunrise <= hour <= sunset
    p = -4.0 / 3.0 if is_day else -2.0 / 3.0
    return cn2_ref * (height_link_ag / CN2_REFERENCE_HEIGHT) ** p


def cn2_hufnagel_valley(height_ag: float, rms_wind: float = 21.0,
                        cn2_ground: float = 1.7e-14) -> float:
    """Hufnagel-Valley profile [AP05, HV5/7] for SLANT PATHS.

    Cn²(h) = 5.94e-53*(v/27)^2*h^10*e^(-h/1000) + 2.7e-16*e^(-h/1500)
             + A*e^(-h/100), h in m ABOVE GROUND, A = surface Cn².

    Kept separate from the horizontal case, which uses
    :func:`cn2_horizontal_link` instead.

    Args:
        height_ag: height ABOVE GROUND LEVEL [m].
        rms_wind: rms wind speed [m/s]; 21 for canonical HV5/7.
        cn2_ground: surface Cn² [m^(-2/3)]; 1.7e-14 for canonical HV5/7.

    Returns:
        float: Cn² [m^(-2/3)] at that height.
    """
    h = height_ag
    return (5.94e-53 * (rms_wind / 27.0) ** 2 * h ** 10 * math.exp(-h / 1000.0)
            + 2.7e-16 * math.exp(-h / 1500.0)
            + cn2_ground * math.exp(-h / 100.0))


# ===========================================================================
# 3. Rain
# ===========================================================================
# Complex refractive index of pure water, Hale & Querry (1973) [HQ73].
# Replaces Quan & Fry (SEA water, 400-700 nm, real part only).
_HALE_QUERRY = {  # λ [m] : (n, k)
    550e-9:  (1.333, 1.96e-9),
    780e-9:  (1.329, 1.43e-7),
    850e-9:  (1.327, 2.93e-7),
    1064e-9: (1.324, 1.20e-6),
    1310e-9: (1.321, 1.20e-4),
    1550e-9: (1.318, 9.86e-5),
}


def water_refractive_index(wavelength: float) -> complex:
    """Complex refractive index m = n - i*k of pure water.

    Logarithmic interpolation of the [HQ73] anchors on the absorption part.

    Args:
        wavelength: wavelength [m].

    Returns:
        complex: m = n - i*k at that wavelength.
    """
    lams = sorted(_HALE_QUERRY)
    if wavelength <= lams[0]:
        n, k = _HALE_QUERRY[lams[0]]
        return complex(n, -k)
    if wavelength >= lams[-1]:
        n, k = _HALE_QUERRY[lams[-1]]
        return complex(n, -k)
    for lo, hi in zip(lams, lams[1:]):
        if lo <= wavelength <= hi:
            t = (wavelength - lo) / (hi - lo)
            n = _HALE_QUERRY[lo][0] + t * (_HALE_QUERRY[hi][0] - _HALE_QUERRY[lo][0])
            k = math.exp(math.log(_HALE_QUERRY[lo][1])
                         + t * (math.log(_HALE_QUERRY[hi][1])
                                - math.log(_HALE_QUERRY[lo][1])))
            return complex(n, -k)


def terminal_velocity_rain(diameter: float) -> float:
    """Terminal velocity v(D) [m/s] of raindrops, Atlas et al. (1973).

        v(D) = 9.65 - 10.3*exp(-600*D)   [D in m; valid 0.6-5.8 mm]

    Stokes' law v = 2r^2*rho*g/(9*mu) is NOT usable here: it holds only for
    Re<<1 (D <~ 0.1 mm) and gives 118.5 m/s at D = 2 mm, against 6.2 m/s
    from Atlas and 6.5 m/s measured by Gunn-Kinzer.

    Args:
        diameter: drop diameter [m].

    Returns:
        float: terminal velocity [m/s], floored at 0.05 m/s.
    """
    if diameter < 0.6e-3 or diameter > 5.8e-3:
        warnings.warn("terminal_velocity_rain: D outside 0.6-5.8 mm; "
                      "extrapolating [Atl73].", RuntimeWarning)
    return max(9.65 - 10.3 * math.exp(-600.0 * diameter), 0.05)


def rain_extinction_marshall_palmer(precipitation_rate: float,
                                    Q_ext: float = 2.0) -> float:
    """Rain extinction coefficient beta [1/m] -- theoretical model.

    Marshall-Palmer drop-size distribution [MP48]:
        N(D) = N0*e^(-Lambda*D), N0 = 8e6 m^-4, Lambda = 4100*R^(-0.21) m^-1
        (R in mm/h -- converted internally from the SI m/s)
    Extinction: beta = (pi/4)*Q_ext*int D^2 N(D) dD = (pi/2)*Q_ext*N0/Lambda^3

    Q_ext = 2 is exact to <1% for x = pi*D/lambda >~ 10^3 (extinction
    paradox, van de Hulst 1957), which covers any raindrop at optical
    wavelengths; to validate against Mie theory use
    :func:`water_refractive_index` with miepython and Q_EXT (not Q_scat:
    water ABSORBS in the NIR, see [HQ73]).

    Cross-check against the empirical [Carb98] form A = 1.076*R^0.67 dB/km:
    at R = 12.5 mm/h this model gives 7.8 dB/km against 5.9 dB/km, i.e. it
    slightly overestimates the optical extinction. A monodisperse
    distribution combined with Stokes' law would underestimate it by ~18x.

    Args:
        precipitation_rate: rain rate [m/s].
        Q_ext: extinction efficiency; 2.0 in the geometric-optics limit.

    Returns:
        float: extinction coefficient [1/m]; 0 without rain.
    """
    if precipitation_rate <= 0:
        return 0.0
    R_mmh = precipitation_rate * 3.6e6          # m/s -> mm/h (SI conversion)
    N0 = 8.0e6                                   # m^-4  [MP48]
    Lam = 4100.0 * R_mmh ** (-0.21)              # m^-1  [MP48]
    return (math.pi / 2.0) * Q_ext * N0 / Lam ** 3


def rain_attenuation_carbonneau(precipitation_rate: float) -> float:
    """Rain extinction beta [1/m] from Carbonneau's empirical form [Carb98].

    A[dB/km] = 1.076*R^0.67 (R in mm/h); beta = A/(4.343e3). Offered as an
    alternative to :func:`rain_extinction_marshall_palmer`.

    Args:
        precipitation_rate: rain rate [m/s].

    Returns:
        float: extinction coefficient [1/m]; 0 without rain.
    """
    if precipitation_rate <= 0:
        return 0.0
    R_mmh = precipitation_rate * 3.6e6
    return 1.076 * R_mmh ** 0.67 / (10.0 * math.log10(math.e)) / 1000.0


# ===========================================================================
# 4. Fog/aerosols (Kim) and phase noise -- SI interfaces
# ===========================================================================
def fog_extinction_kim(atm_visibility: float, wavelength: float) -> float:
    """Fog/haze extinction coefficient beta [1/m], Kim's model [Kim01].

    Args:
        atm_visibility: meteorological (atmospheric) visibility [m]; not to
            be confused with the COW interferometer visibility.
        wavelength: wavelength [m].

    Returns:
        float: extinction coefficient [1/m]. V is converted to km and
        lambda to nm internally, as published in [Kim01].
    """
    V_km = atm_visibility * 1e-3                 # m -> km (SI conversion)
    lam_nm = wavelength * 1e9                    # m -> nm (SI conversion)
    if V_km > 50:
        q = 1.6
    elif V_km > 6:
        q = 1.3
    elif V_km > 1:
        q = 0.16 * V_km + 0.34
    elif V_km > 0.5:
        q = V_km - 0.5
    else:
        q = 0.0
    return (3.91 / V_km) * (lam_nm / 550.0) ** (-q) * 1e-3   # 1/km → 1/m


def phase_noise(wavelength: float, C_n2: float, height_ag: float) -> float:
    """Square root of the phase variance per metre [rad/sqrt(m)].

    From [AP05] Eq.(75) p.289. Use ONLY when
    :class:`AtmosphericPhaseProcess` is not in play, otherwise the
    turbulence is counted twice.

    Args:
        wavelength: wavelength [m].
        C_n2: refractive-index structure parameter [m^(-2/3)].
        height_ag: link height ABOVE GROUND LEVEL [m].

    Returns:
        float: phase-noise coefficient [rad/sqrt(m)].
    """
    k = 2.0 * math.pi / wavelength
    K0 = 2.0 * math.pi / outer_scale(height_ag)
    return math.sqrt(0.78 * C_n2 * k ** 2 * K0 ** (-5.0 / 3.0))


# ===========================================================================
# 5. Total FSO channel loss
# ===========================================================================
def channel_FSO_loss(distance: float, wavelength: float, atm_visibility: float,
                     receiver_radius: float, pressure: float,
                     temperature: float, w_0: float, R_0: float,
                     friction_velocity: float, height_ag: float,
                     precipitation_rate: float = 0.0,
                     C_n2: Optional[float] = None,
                     C_T2: Optional[float] = None,
                     Q_ext_rain: float = 2.0,
                     rain_model: str = "marshall_palmer") -> float:
    """Long-term MEAN loss fraction of the FSO channel, in [0, 1].

    Structure ([1] Choudhury & Nandi 2024; [GP22]):
        eta_total = eta_fog * eta_rain * eta_turb,  loss = 1 - eta_total
        eta_turb  = 1 - exp(-2a^2/w_LT^2)
    The turbulence term is the LONG-TERM collection efficiency; the
    instantaneous scintillation must be handled by a separate stochastic
    process (see :class:`AtmosphericPhaseProcess` for the phase part).

    Args:
        distance            [m]     link length
        wavelength          [m]     e.g. 780e-9
        atm_visibility      [m]     meteorological (atmospheric) visibility
        receiver_radius     [m]     receiver aperture radius
        pressure            [Pa]    atmospheric pressure
        temperature         [K]     air temperature
        w_0                 [m]     initial Gaussian beam waist
        R_0                 [m]     initial curvature radius (math.inf if collimated)
        friction_velocity   [m/s]   u*, strictly in m/s
        height_ag           [m]     link height ABOVE GROUND LEVEL
        precipitation_rate  [m/s]   1 mm/h = 2.78e-7 m/s
        C_n2                [m^(-2/3)] derived from C_T2 [AP05] when None
        C_T2                [K^2 m^(-2/3)] used only when C_n2 is None
        Q_ext_rain          extinction efficiency of the rain model
        rain_model          "marshall_palmer" (theoretical) | "carbonneau" (empirical)

    Returns:
        float: loss fraction in [0, 1].

    Raises:
        ValueError: if neither C_n2 nor C_T2 is given, or if rain_model is
            not one of the two supported names.
    """
    k = 2.0 * math.pi / wavelength
    l0 = inner_scale(temperature, pressure, friction_velocity, height_ag)

    # --- fog/aerosols (Kim) --------------------------------------------
    eta_fog = math.exp(-fog_extinction_kim(atm_visibility, wavelength) * distance)

    # --- turbulence: long-term beam spreading [GP22, AP05] -------------
    if C_n2 is None:
        if C_T2 is None:
            raise ValueError("Either C_n2 or C_T2 must be provided.")
        # [AP05]/Murty: Cn = 77.6e-6*(P[mbar]/T^2)*(1+7.53e-3/lam_um^2)*C_T
        P_mbar = pressure * 1e-2                 # Pa -> mbar (SI conversion)
        lam_um = wavelength * 1e6                # m -> um  (SI conversion)
        C_n2 = ((77.6e-6 * P_mbar / temperature ** 2) ** 2
                * (1.0 + 0.00753 / lam_um ** 2) ** 2 * C_T2)

    Z_R = math.pi * w_0 ** 2 / wavelength                       # Rayleigh
    sigma_R2 = 1.23 * C_n2 * k ** (7.0 / 6.0) * distance ** (11.0 / 6.0)
    w_z2 = w_0 ** 2 * ((1.0 - distance / R_0) ** 2 + (distance / Z_R) ** 2)

    # Regime criterion via the plane-wave coherence length
    # rho0 = (1.46*Cn^2*k^2*L)^(-3/5) compared to l0 [AP05, GP22].
    rho0 = (1.46 * C_n2 * k ** 2 * distance) ** (-3.0 / 5.0)
    Lambda_par = 2.0 * distance / (k * w_z2)
    if rho0 < l0:      # moderate-strong (the smallest cell l0 dominates)
        w_lt2 = w_z2 * (1.0 + 0.74 * (4.0 / 3.0) * sigma_R2
                        * ((35.05 * distance / (k * l0 ** 2)) ** (1.0 / 6.0))
                        * Lambda_par)
    else:              # weak-moderate
        w_lt2 = w_z2 * (1.0 + 1.63 * sigma_R2 ** (6.0 / 5.0) * Lambda_par)
    eta_turb = 1.0 - math.exp(-2.0 * receiver_radius ** 2 / w_lt2)

    # --- rain ----------------------------------------------------------
    if rain_model == "marshall_palmer":
        beta_rain = rain_extinction_marshall_palmer(precipitation_rate, Q_ext_rain)
    elif rain_model == "carbonneau":
        beta_rain = rain_attenuation_carbonneau(precipitation_rate)
    else:
        raise ValueError(f"unknown rain_model: {rain_model!r}")
    eta_rain = math.exp(-beta_rain * distance)

    return 1.0 - eta_fog * eta_rain * eta_turb


# ===========================================================================
# 6. Atmospheric phase process
# ===========================================================================
class AtmosphericPhaseProcess:
    """Temporally correlated piston phase phi(t) [AP05] Eqs. 75, 108, 110.

    The process is pre-generated by spectral synthesis over the whole
    simulation window and then sampled by :meth:`phase_at`, so repeated
    queries are cheap and deterministic. All interfaces are SI (wavelength
    in METRES).
    """

    def __init__(self, duration_s: float, dt_s: float, wavelength: float,
                 C_n2: float, distance: float, outer_scale_m: float,
                 wind_speed_perp: float, seed: Optional[int] = None) -> None:
        """Pre-generate the phase realisation covering the whole run.

        Args:
            duration_s: length of the window to synthesise [s].
            dt_s: sampling step of the synthesis [s].
            wavelength: optical wavelength [m].
            C_n2: refractive-index structure parameter [m^(-2/3)].
            distance: propagation length [m].
            outer_scale_m: turbulence outer scale L0 [m].
            wind_speed_perp: transverse wind speed [m/s], which sets the
                atmospheric coherence time tau_atm = 1/(kappa0*V_perp).
            seed: seed of the synthesis RNG; the realisation is fully
                determined by it, which is what makes a run reproducible.

        Raises:
            ValueError: if the sampling step does not satisfy
                0 < dt_s < duration_s.
        """
        if not (0 < dt_s < duration_s):
            raise ValueError("0 < dt_s < duration_s is required.")
        self.duration_s = float(duration_s)
        self.dt_s = float(dt_s)
        self.C_n2 = float(C_n2)
        self.distance = float(distance)
        self._k = 2.0 * math.pi / float(wavelength)          # lambda in m (SI)
        self._kappa_0 = 2.0 * math.pi / float(outer_scale_m)
        self.wind_speed_perp = float(wind_speed_perp)
        self.tau_atm = 1.0 / (self._kappa_0 * self.wind_speed_perp)
        self.theoretical_variance = (0.78 * self.C_n2 * self._k ** 2
                                     * self.distance
                                     * self._kappa_0 ** (-5.0 / 3.0))
        self._values, self._N = self._synthesize(seed)
        self._clamp_warned = False

    def _psd_one_sided(self, omega: np.ndarray) -> np.ndarray:
        """One-sided temporal power spectral density of the piston phase.

        Args:
            omega: angular frequencies [rad/s].

        Returns:
            np.ndarray: PSD [rad^2 s/rad] at those frequencies.
        """
        num = (5.82 * self.C_n2 * self._k ** 2 * self.distance
               * self.wind_speed_perp ** (5.0 / 3.0))
        den = (omega ** 2
               + (self._kappa_0 * self.wind_speed_perp) ** 2) ** (4.0 / 3.0)
        return num / den

    def _synthesize(self, seed):
        """Synthesise the phase time series from the PSD by inverse FFT.

        The series is rescaled to the theoretical variance so that the
        finite synthesis window does not bias the phase excursion.

        Args:
            seed: seed of the RNG drawing the spectral coefficients.

        Returns:
            tuple: (samples array, number of samples N).
        """
        rng = np.random.default_rng(seed)
        N = int(math.ceil(self.duration_s / self.dt_s))
        N += N % 2
        omegas = 2.0 * math.pi * np.fft.rfftfreq(N, d=self.dt_s)
        amp = np.sqrt(N * self._psd_one_sided(omegas) / (2.0 * self.dt_s))
        amp[0] *= math.sqrt(2.0)
        amp[-1] *= math.sqrt(2.0)
        z_re = rng.standard_normal(len(omegas))
        z_im = rng.standard_normal(len(omegas))
        z_im[0] = 0.0
        z_im[-1] = 0.0
        x = np.fft.irfft(amp * (z_re + 1j * z_im) / math.sqrt(2.0), n=N)
        var = float(np.var(x))
        if var > 0 and self.theoretical_variance > 0:
            x *= math.sqrt(self.theoretical_variance / var)
        return x, N

    def sample(self, time_ps: float) -> float:
        """Phase phi at simulation time ``time_ps``, linearly interpolated.

        Args:
            time_ps: simulation instant [ps], the kernel time unit.

        Returns:
            float: the piston phase [rad]; clamped (with a one-shot
            RuntimeWarning) outside the pre-generated window.
        """
        t_s = float(time_ps) * 1e-12
        if t_s <= 0.0:
            return float(self._values[0])
        if t_s >= (self._N - 1) * self.dt_s:
            if not self._clamp_warned:
                warnings.warn("AtmosphericPhaseProcess: query beyond the "
                              "pre-generated window; value clamped.",
                              RuntimeWarning)
                self._clamp_warned = True
            return float(self._values[-1])
        idx = t_s / self.dt_s
        i = int(idx)
        f = idx - i
        return float((1 - f) * self._values[i] + f * self._values[i + 1])


def make_atmospheric_phase_process(distance, timeline_stop_time_ps, ls_params,
                                   loss_parameters, seed=None):
    """Build an AtmosphericPhaseProcess covering a whole timeline.

    The sampling step defaults to 2% of the atmospheric coherence time,
    clipped to [1 us, 1 ms], and can be overridden with the
    ``phase_dt_s`` entry of ``loss_parameters``.

    Args:
        distance: propagation length of the segment [m].
        timeline_stop_time_ps: simulation horizon to cover [ps].
        ls_params: light-source parameters; ``wavelength`` in m (SI).
        loss_parameters: atmospheric parameters; ``height_ag`` in m,
            ``C_n2`` and ``wind_speed_perp`` in SI.
        seed: seed of the synthesis, forwarded for reproducibility.

    Returns:
        AtmosphericPhaseProcess: ready to be sampled by the COW channel.
    """
    L0 = outer_scale(loss_parameters["height_ag"])
    V_perp = loss_parameters["wind_speed_perp"]
    tau_atm = L0 / (2.0 * math.pi * V_perp)
    dt_s = float(loss_parameters.get(
        "phase_dt_s", max(min(0.02 * tau_atm, 1.0e-3), 1.0e-6)))
    return AtmosphericPhaseProcess(
        duration_s=timeline_stop_time_ps * 1e-12, dt_s=dt_s,
        wavelength=ls_params["wavelength"], C_n2=loss_parameters["C_n2"],
        distance=distance, outer_scale_m=L0,
        wind_speed_perp=V_perp, seed=seed)

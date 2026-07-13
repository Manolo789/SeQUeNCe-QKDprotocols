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

Substitui o chaveamento binário dia/noite de scenarios.py
(B_sky = 1,5e-6 ↔ 1,5e-2 W·m⁻²·nm⁻¹·sr⁻¹) por um modelo físico contínuo.

CADEIA DO MODELO (céu claro):
  1. Posição solar (declinação, equação do tempo, ângulo horário) —
     séries de Fourier de Spencer (1971), como no algoritmo solar do NOAA;
     ver também Meeus, "Astronomical Algorithms" (1998).
  2. Massa de ar relativa — Kasten & Young (1989), válida até o horizonte.
  3. Irradiância difusa espectral no plano horizontal — componentes de
     Rayleigh e de aerossol do modelo SPCTRAL2 de Bird & Riordan (1986),
     avaliadas no comprimento de onda do enlace.  Nas janelas atmosféricas
     usadas em QKD (770–860 nm, 1550 nm) as transmitâncias de O3, H2O e
     gases uniformes são ≈1 e são omitidas (hipótese documentada; para
     λ dentro de bandas de absorção, usar SPCTRAL2 completo ou MODTRAN).
     Turbidez de aerossol pela lei de Ångström τ_a = β·λ_µm^(−α)
     (Ångström 1964; valores típicos de β em Iqbal, "An Introduction to
     Solar Radiation", 1983: 0,05 limpo / 0,1 médio / 0,2 túrbido).
  4. Radiância do céu: hipótese de céu isotrópico L = E_dif/π
     (Liou, "An Introduction to Atmospheric Radiation", 2002, §1.1;
     aproximação de primeira ordem — o céu claro real varia com o ângulo
     de espalhamento em relação ao Sol, tipicamente ±(2–5)×).
  5. Crepúsculo (−18° < h < 0°): decaimento log-linear entre L(h=0) e o
     piso noturno em h=−18° — consistente com as ~3–4 décadas de queda da
     radiância zenital medidas por Rozenberg, "Twilight: A Study in
     Atmospheric Optics" (1966); ver também Patat, A&A 400, 1183 (2003).
  6. Noite (h ≤ −18°): piso configurável.  Valores de referência
     (Er-long Miao et al., New J. Phys. 7, 215 (2005); Bourgoin et al.,
     New J. Phys. 15, 023006 (2013); Pirandola, PRR 3, 023130 (2021)):
       lua cheia, céu claro : ~1,5e-3 W·m⁻²·µm⁻¹·sr⁻¹ = 1,5e3  SI
       lua nova,  céu claro : ~1,5e-6 W·m⁻²·µm⁻¹·sr⁻¹ = 1,5e0  SI

UNIDADES: SI estrito.  Radiância espectral em W·m⁻²·m⁻¹·sr⁻¹
(por METRO de comprimento de onda).  Conversões:
  1 W·m⁻²·nm⁻¹·sr⁻¹ = 1e9 SI   |   1 W·m⁻²·µm⁻¹·sr⁻¹ = 1e6 SI
O antigo B_sky=1,5e-2 (por nm) do fork equivale a 1,5e7 SI.
"""

import math
import warnings
from datetime import datetime, timezone

H_PLANCK = 6.62607015e-34   # J·s
C_LIGHT = 2.99792458e8      # m/s

# ---------------------------------------------------------------------------
# Irradiância solar extraterrestre E0(λ) — âncoras do espectro ASTM E490
# (AM0), em W·m⁻²·m⁻¹ (SI). Interpolação linear entre âncoras.
# ---------------------------------------------------------------------------
_E0_AM0 = {  # λ [m] : E0 [W·m⁻²·m⁻¹]
    400e-9: 1.60e9, 500e-9: 1.95e9, 550e-9: 1.86e9, 650e-9: 1.55e9,
    780e-9: 1.17e9, 850e-9: 0.97e9, 1000e-9: 0.75e9, 1064e-9: 0.65e9,
    1310e-9: 0.37e9, 1550e-9: 0.265e9,
}


def extraterrestrial_irradiance(wavelength: float) -> float:
    """E0(λ) [W·m⁻²·m⁻¹] no topo da atmosfera (âncoras ASTM E490)."""
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
# 1. Posição solar — Spencer (1971) / NOAA
# ---------------------------------------------------------------------------
def solar_elevation(when_utc: datetime, latitude: float,
                    longitude: float) -> float:
    """Elevação solar h [rad] (negativa abaixo do horizonte).

    Args:
        when_utc : datetime COM tzinfo UTC (ou naive interpretado como UTC)
        latitude : [rad], positivo Norte
        longitude: [rad], positivo Leste
    Referências: Spencer, Search 2, 172 (1971); NOAA Solar Calculator;
    Meeus (1998). Precisão ~0,01 rad — suficiente para radiometria.
    """
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    when_utc = when_utc.astimezone(timezone.utc)
    doy = when_utc.timetuple().tm_yday
    frac_h = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    # ângulo do ano [rad]
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (frac_h - 12.0) / 24.0)
    # declinação solar [rad] — Spencer (1971)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    # equação do tempo [min] — Spencer (1971)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g)
                    - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    # hora solar verdadeira e ângulo horário
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
# 2. Radiância difusa de céu claro — SPCTRAL2 simplificado (Bird & Riordan)
# ---------------------------------------------------------------------------
def _rayleigh_od(wavelength: float, pressure: float) -> float:
    """Profundidade óptica de Rayleigh τ_R(λ) na vertical, Bird & Riordan
    (1986): τ_R = (P/P0)/[λ⁴·(115,6406 − 1,335/λ²)], λ em µm."""
    lam_um = wavelength * 1e6                    # m → µm (fórmula publicada)
    return (pressure / 101325.0) / (lam_um ** 4
                                    * (115.6406 - 1.335 / lam_um ** 2))


def clear_sky_radiance(wavelength: float, elevation: float,
                       pressure: float = 101325.0,
                       angstrom_beta: float = 0.10,
                       angstrom_alpha: float = 1.3,
                       ssa: float = 0.90, asym_g: float = 0.65) -> float:
    """Radiância espectral do céu claro L [W·m⁻²·m⁻¹·sr⁻¹] para h > 0.

    Componentes difusas de Rayleigh (I_r) e de aerossol (I_a) do SPCTRAL2
    [Bird & Riordan 1986, Eqs. 3-9 a 3-13], com T_O3=T_H2O=T_gás=1
    (janelas atmosféricas de QKD) e sem termo de reflexão do solo;
    L = (I_r + I_a)/π  (céu isotrópico, Liou 2002).

    Parâmetros de aerossol: lei de Ångström τ_a=β·λ_µm^(−α) [Ångström
    1964; Iqbal 1983], albedo de espalhamento simples ω0 (ssa) e
    assimetria g típicos continentais [Bird & Riordan 1986].
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
    T_aa = math.exp(-(1.0 - ssa) * tau_a * M)     # absorção do aerossol
    T_as = math.exp(-ssa * tau_a * M)             # espalhamento do aerossol

    # Fração espalhada para baixo pelo aerossol, F_s [Bird & Riordan 1986]
    alg = math.log(1.0 - asym_g)
    afs = alg * (1.459 + alg * (0.1595 + 0.4129 * alg))
    bfs = alg * (0.0783 + alg * (-0.3824 - 0.5874 * alg))
    F_s = 1.0 - 0.5 * math.exp((afs + bfs * cosZ) * cosZ)

    I_r = E0 * cosZ * T_aa * 0.5 * (1.0 - T_r ** 0.95)
    I_a = E0 * cosZ * T_aa * (T_r ** 1.5) * (1.0 - T_as) * F_s
    return (I_r + I_a) / math.pi


# ---------------------------------------------------------------------------
# 3. B_sky contínuo: dia + crepúsculo + noite
# ---------------------------------------------------------------------------
B_NIGHT_NEW_MOON = 1.5e0    # W·m⁻²·m⁻¹·sr⁻¹  (= 1,5e-6 µm⁻¹) [Miao05/Bourgoin13]
B_NIGHT_FULL_MOON = 1.5e3   # W·m⁻²·m⁻¹·sr⁻¹  (= 1,5e-3 µm⁻¹) [Pirandola21]
_TWILIGHT_END = math.radians(-18.0)   # crepúsculo astronômico


def b_sky(wavelength: float, elevation: float,
          pressure: float = 101325.0,
          angstrom_beta: float = 0.10,
          b_night: float = B_NIGHT_FULL_MOON,
          cloud_factor: float = 1.0) -> float:
    """Radiância espectral do céu B_sky [W·m⁻²·m⁻¹·sr⁻¹] vs elevação solar.

    Regimes:
      h > 0        : céu claro SPCTRAL2 (clear_sky_radiance), ×cloud_factor
                     (céu encoberto aumenta a radiância difusa ~10×;
                     Bourgoin et al. 2013, Tab. 2)
      −18° < h ≤ 0 : interpolação log-linear entre L(h→0⁺) e b_night
                     [Rozenberg 1966: ~3–4 décadas em 18°]
      h ≤ −18°     : b_night (lua nova/cheia — B_NIGHT_*)
    """
    L_day_horizon = clear_sky_radiance(
        wavelength, math.radians(0.5), pressure, angstrom_beta) * cloud_factor
    if elevation > 0:
        L = clear_sky_radiance(wavelength, elevation, pressure,
                               angstrom_beta) * cloud_factor
        return max(L, b_night)
    if elevation > _TWILIGHT_END:
        f = elevation / _TWILIGHT_END            # 0 no pôr do sol → 1 no fim
        logL = ((1.0 - f) * math.log10(max(L_day_horizon, b_night))
                + f * math.log10(b_night))
        return 10.0 ** logL
    return b_night


def b_sky_at(when_utc: datetime, latitude: float, longitude: float,
             wavelength: float, **kwargs) -> float:
    """Conveniência: B_sky [SI] a partir de instante UTC e coordenadas
    [rad] — para uso direto no diurnal_profile() de scenarios.py."""
    h = solar_elevation(when_utc, latitude, longitude)
    return b_sky(wavelength, h, **kwargs)


# ---------------------------------------------------------------------------
# 4. Fótons de fundo por modo — CORREÇÃO DO TÓPICO 6
# ---------------------------------------------------------------------------
def n_background(wavelength: float, filter_bandwidth: float,
                 detection_gate: float, fov_solid_angle: float,
                 receiver_radius: float, B_sky_si: float) -> float:
    """n_B — fótons de fundo por modo de detecção (adimensional).

    Pirandola, PRR 3, 023130 (2021), Eq. (32):
        n_B = π·λ·Γ_R·B_sky/(h·c),   Γ_R = Δλ·Δt·Ω_fov·a_R²

    Args — TODOS EM SI:
        wavelength       [m]
        filter_bandwidth [m]   Δλ do filtro espectral (1 nm → 1e-9)
        detection_gate   [s]   Δt — CORREÇÃO (tópico 6): é a JANELA DE
                               ACEITAÇÃO temporal do clique (gate casado ao
                               pulso/à resolução do detector), e NÃO o tempo
                               morto 1/count_rate. Com os parâmetros do fork
                               (time_resolution=1 ns, 1/count_rate=50 ns),
                               o uso do tempo morto superestimava o fundo
                               em 50×. Recomenda-se
                               Δt = max(time_resolution, largura_do_pulso).
        fov_solid_angle  [sr]  Ω do campo de visão do receptor
        receiver_radius  [m]   a_R
        B_sky_si         [W·m⁻²·m⁻¹·sr⁻¹]  (saída de b_sky/b_sky_at)
    """
    gamma_R = (filter_bandwidth * detection_gate * fov_solid_angle
               * receiver_radius ** 2)
    return math.pi * wavelength * gamma_R * B_sky_si / (H_PLANCK * C_LIGHT)


def detection_gate_from_detector(time_resolution_ps: float,
                                 pulse_width_s: float = 0.0) -> float:
    """Δt [s] recomendado para n_background: gate casado à resolução
    temporal do detector (parâmetro `time_resolution` do SeQUeNCe, em ps)
    ou à largura do pulso, o que for maior."""
    return max(time_resolution_ps * 1e-12, pulse_width_s)

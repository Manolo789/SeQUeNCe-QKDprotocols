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

CONVENÇÃO DE UNIDADES (SI estrito em todas as interfaces públicas):
    comprimento .......... m        temperatura ........ K
    tempo ................ s        pressão ............ Pa
    velocidade ........... m/s      viscosidade din. ... Pa·s
    comprimento de onda .. m        taxa de precipitação m/s
    Cn² .................. m^(-2/3) radiância espectral  W·m⁻²·m⁻¹·sr⁻¹
Conversões para fórmulas empíricas publicadas em outras unidades (ex.:
visibilidade em km no modelo de Kim, R em mm/h em Marshall–Palmer) são
feitas INTERNAMENTE, com comentário explícito no ponto de conversão.

CORREÇÕES EM RELAÇÃO AO loss.py ORIGINAL:
  -- Atenuação por Chuva: a lei de Stokes (Re≪1) é inválida para gotas de chuva
       (Re~10²–10³; dava v_t≈118 m/s p/ D=2 mm vs ~6,5 m/s medidos).
       Substituída por distribuição de tamanhos de Marshall–Palmer (1948)
       + Q_ext=2 (paradoxo da extinção, x=πD/λ≳10³) com forma fechada,
       velocidade terminal empírica de Atlas et al. (1973) e verificação
       cruzada com a forma empírica de Carbonneau.
  -- Cn²: (i) removida a extrapolação da lei de camada superficial h^(-4/3)
       com o termo exponencial do perfil Hufnagel–Valley (inflação ~37×); 
       (ii) enlaces horizontais próximos ao solo agora usam o valor de 
       SUPERFÍCIE escalado por h^(-4/3) (diurno instável) ou h^(-2/3) 
       (noturno), com h ACIMA DO SOLO; o perfil HV fica em função 
       separada para trajetos inclinados, também com h acima do solo.
  -- inner_scale: eliminada a ambiguidade cm/s vs m/s de u*;
       o critério de transição fraco↔forte usa o comprimento de coerência
       de onda plana com o fator 1,46: ρ0=(1,46·Cn²·k²·L)^(-3/5).
  -- polarization_fidelity() removida por decisão de projeto: a
       fidelidade de polarização deve ser tomada como 1.0 no simulador
       (turbulência preserva polarização em modo único; ver Fedrizzi et
       al., Nat. Phys. 5, 389 (2009), enlace de 144 km com erro <1%). A
       formulação completa da polarization_fidelity() depende do estudo do
       efeito de aerossóis na degradação do estado de polarização.

REFERÊNCIAS PRINCIPAIS:
  [AP05]  Andrews & Phillips, "Laser Beam Propagation Through Random
          Media", 2ª ed., SPIE (2005).
  [SK92]  Sadot & Kopeika, Opt. Eng. 31, 200 (1992) — modelo macroescala
          de Cn² (T em Kelvin).
  [Ben04] Bendersky, Kopeika & Blaunstein, Appl. Opt. 43, 4070 (2004).
  [Wyn71] Wyngaard, Izumi & Collins, J. Opt. Soc. Am. 61, 1646 (1971) —
          Cn² ∝ h^(-4/3) (instável) e h^(-2/3) (neutro/estável).
  [MP48]  Marshall & Palmer, J. Meteor. 5, 165 (1948) — N(D)=N0·e^(-ΛD).
  [Atl73] Atlas, Srivastava & Sekhon, Rev. Geophys. 11, 1 (1973) —
          v(D)=9,65−10,3·e^(−600·D) m/s (D em m; 0,6–5,8 mm).
  [GK49]  Gunn & Kinzer, J. Meteor. 6, 243 (1949) — v_t medidas.
  [Carb98] Carbonneau & Wisely, Proc. SPIE 3232 (1998) —
          A[dB/km]=1,076·R^0,67 (R em mm/h).
  [HQ73]  Hale & Querry, Appl. Opt. 12, 555 (1973) — índice complexo da
          água 0,2–200 µm (substitui Quan & Fry, que vale só para água do
          mar no visível e ignora a parte imaginária).
  [Kim01] Kim, McArthur & Korevaar, Proc. SPIE 4214 (2001) — atenuação
          por visibilidade.
  [HV]    Hufnagel–Valley 5/7, ver [AP05] cap. 12.
  [GP22]  Ghalaii & Pirandola, Commun. Phys. 5, 38 (2022).
"""

from typing import Optional
import math
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Constantes físicas (SI)
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8        # m/s
H_PLANCK = 6.62607015e-34     # J·s
R_GAS = 8.314462618           # J/(mol·K)
M_AIR = 0.0289645             # kg/mol
KAPPA_VK = 0.4                # constante de von Kármán
G_STD = 9.80665               # m/s²


# ===========================================================================
# 1. Micrometeorologia (SI)
# ===========================================================================
def f_velocity(wind_speed: float, T_classification: int, height_ag: float) -> float:
    """Velocidade de atrito u* [m/s] pelo perfil logarítmico neutro.

    u* = κ·U(z) / ln(z/z0)   [Holton 2004; WMO Guide 2024, tabela de z0]

    Args:
        wind_speed: velocidade do vento na altura height_ag [m/s]
        T_classification: classe de terreno WMO (1–8)
        height_ag: altura da medição ACIMA DO SOLO [m]
    Returns:
        u* [m/s]  (SI — sem fator 100; ver correção [C5])
    """
    z0 = {1: 0.0002, 2: 0.005, 3: 0.03, 4: 0.10,
          5: 0.25, 6: 0.5, 7: 1.0, 8: 2.0}[T_classification]
    if height_ag <= z0:
        raise ValueError("height_ag deve ser maior que z0 do terreno.")
    return KAPPA_VK * wind_speed / math.log(height_ag / z0)


def viscosity_sutherland(temperature: float) -> float:
    """Viscosidade dinâmica do ar [Pa·s] pela equação de Sutherland (1893).

    μ = μ0·((T0+S)/(T+S))·(T/T0)^(3/2), μ0=1,716e-5 Pa·s, T0=273,15 K,
    S=110,4 K.  (Original retornava CGS, g·cm⁻¹·s⁻¹ = poise·10.)
    """
    return 1.716e-5 * ((273.15 + 110.4) / (temperature + 110.4)) \
        * (temperature / 273.15) ** 1.5


def air_density(pressure: float, temperature: float) -> float:
    """Densidade do ar [kg/m³] pelo gás ideal.  pressure em Pa, T em K."""
    return pressure * M_AIR / (R_GAS * temperature)


def outer_scale(height_ag: float) -> float:
    """Escala externa L0 [m] da turbulência.  height_ag ACIMA DO SOLO [m].

    Parametrização de [AP05] p.483 / Lukin (2005).  (Original recebia cm.)
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
    """Escala interna l0 [m] (Tatarskii), tudo em SI.  [AP05] p.57–82.

    l0 = 7,4·(ν³/ε)^(1/4);  ν = μ/ρ;  ε = u*³/(κ·h)  (camada superficial).

    CORREÇÃO [C5]: friction_velocity em m/s (o original documentava cm/s e
    era alimentado ora em m/s (scenarios.py), ora em cm/s (QKD_Extension),
    produzindo l0 de 119 mm vs 3,8 mm).  height_ag ACIMA DO SOLO em m.
    """
    if viscosity is None:
        viscosity = viscosity_sutherland(temperature)
    nu = viscosity / air_density(pressure, temperature)          # m²/s
    eps = friction_velocity ** 3 / (KAPPA_VK * height_ag)        # m²/s³
    return 7.4 * (nu ** 3 / eps) ** 0.25


def wind_speed_perp(height_asl: float, ground_wind: float,
                    slew_rate: float = 0.0) -> float:
    """Vento transversal V(h) [m/s], modelo de Bufton [AP05] cap.12 Eq.(3).

    height_asl: altitude [m] (o jato de ~9,4 km é referido ao nível do mar).
    """
    tropo_jet = 30.0 * math.exp(-((height_asl - 9400.0) / 4800.0) ** 2)
    return slew_rate * height_asl + ground_wind + tropo_jet


# ===========================================================================
# 2. Cn²
# ===========================================================================
def _temporal_hour_weight(hour: float, sunrise: float, sunset: float) -> float:
    """Peso W(hora temporal) de Sadot–Kopeika/Bendersky [SK92, Ben04]."""
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
    """Cn² [m^(-2/3)] na ALTURA DE REFERÊNCIA h_ref = 15 m acima do solo.

    Regressão macroescala de [SK92]/[Ben04]:
        Cn² = 3,8e-14·W + 2e-15·T − 2,5e-15·U + 1,2e-15·U² − 8,5e-17·U³
              − 2,8e-15·RH + 2,9e-17·RH² − 1,1e-19·RH³ − 5,3e-13

    Validade [Ben04]: 282 K ≤ T ≤ 308 K; 0 ≤ U ≤ 10 m/s; 14 % ≤ RH ≤ 92 %.
    """
    T, U, RH = temperature, wind_speed, relative_humidity
    if not (282.0 <= T <= 308.0 and 0.0 <= U <= 10.0 and 14.0 <= RH <= 92.0):
        warnings.warn(
            f"cn2_surface_sadot_kopeika: (T={T:.1f} K, U={U:.1f} m/s, "
            f"RH={RH:.0f} %) fora do domínio de validade da regressão "
            "[Ben04]; resultado é extrapolação.", RuntimeWarning)

    W = _temporal_hour_weight(hour, sunrise, sunset)
    cn2 = (3.8e-14 * W
           + 2.0e-15 * T
           - 2.5e-15 * U + 1.2e-15 * U ** 2 - 8.5e-17 * U ** 3
           - 2.8e-15 * RH + 2.9e-17 * RH ** 2 - 1.1e-19 * RH ** 3
           - 5.3e-13)
    if cn2 <= 0:
        warnings.warn("cn2_surface_sadot_kopeika: regressão retornou valor "
                      "não-físico (≤0); usando 1e-17 m^(-2/3). Verifique as "
                      "unidades/domínio das entradas.", RuntimeWarning)
        cn2 = 1e-17
    return cn2

CN2_REFERENCE_HEIGHT = 15.0   # m acima do solo — altura da regressão [Ben04]


def cn2_horizontal_link(height_link_ag: float, hour: float, sunrise: float,
                        sunset: float, temperature: float, wind_speed: float,
                        relative_humidity: float) -> float:
    """Cn² [m^(-2/3)] na altura do ENLACE HORIZONTAL, acima do solo.

    CORREÇÃO [i,ii]: em vez de injetar o valor de superfície (escalado
    por 15^{4/3}) no termo exponencial do perfil Hufnagel–Valley avaliado
    na ALTITUDE ACIMA DO NÍVEL DO MAR — o que aniquilava o termo por
    exp(−h_asl/100) e produzia Cn²~8e-16 ao meio-dia —, aplica-se a lei de
    camada superficial [Wyn71]:

        Cn²(h) = Cn²(h_ref)·(h/h_ref)^p,  p = −4/3 (diurno, instável)
                                          p = −2/3 (noturno, neutro/estável)

    com h = altura ACIMA DO SOLO.  Para o enlace de 8 m do cenário-base
    isso dá ~5e-14 ao meio-dia — dentro da faixa típica diurna junto ao
    solo (1e-14–1e-13), vs 8e-16 do código original.
    """
    cn2_ref = cn2_surface_sadot_kopeika(hour, sunrise, sunset, temperature,
                                        wind_speed, relative_humidity)
    is_day = sunrise <= hour <= sunset
    p = -4.0 / 3.0 if is_day else -2.0 / 3.0
    return cn2_ref * (height_link_ag / CN2_REFERENCE_HEIGHT) ** p


def cn2_hufnagel_valley(height_ag: float, rms_wind: float = 21.0,
                        cn2_ground: float = 1.7e-14) -> float:
    """Perfil Hufnagel–Valley [AP05, HV5/7] para TRAJETOS INCLINADOS.

    Cn²(h) = 5,94e-53·(v/27)²·h¹⁰·e^(−h/1000) + 2,7e-16·e^(−h/1500)
             + A·e^(−h/100),  h em m ACIMA DO SOLO, A = Cn² à superfície.

    Mantida separada do caso horizontal.  Para HV5/7
    canônico: rms_wind=21 m/s, cn2_ground=1,7e-14 m^(-2/3).
    """
    h = height_ag
    return (5.94e-53 * (rms_wind / 27.0) ** 2 * h ** 10 * math.exp(-h / 1000.0)
            + 2.7e-16 * math.exp(-h / 1500.0)
            + cn2_ground * math.exp(-h / 100.0))


# ===========================================================================
# 3. Chuva
# ===========================================================================
# Índice de refração complexo da água pura, Hale & Querry (1973) [HQ73].
# Substitui Quan & Fry (água do MAR, 400–700 nm, só parte real).
_HALE_QUERRY = {  # λ [m] : (n, k)
    550e-9:  (1.333, 1.96e-9),
    780e-9:  (1.329, 1.43e-7),
    850e-9:  (1.327, 2.93e-7),
    1064e-9: (1.324, 1.20e-6),
    1310e-9: (1.321, 1.20e-4),
    1550e-9: (1.318, 9.86e-5),
}


def water_refractive_index(wavelength: float) -> complex:
    """m = n − i·k da água pura em λ [m], interp. log em [HQ73]."""
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
    """Velocidade terminal v(D) [m/s] de gotas de chuva, Atlas et al. (1973):

        v(D) = 9,65 − 10,3·exp(−600·D)   [D em m; válido 0,6–5,8 mm]

    CORREÇÃO: substitui a lei de Stokes v=2r²ρg/(9μ), válida só para
    Re≪1 (D≲0,1 mm).  Para D=2 mm: Stokes → 118,5 m/s; Atlas → 6,2 m/s;
    Gunn–Kinzer (medido) → 6,5 m/s.
    """
    if diameter < 0.6e-3 or diameter > 5.8e-3:
        warnings.warn("terminal_velocity_rain: D fora de 0,6–5,8 mm; "
                      "extrapolação de [Atl73].", RuntimeWarning)
    return max(9.65 - 10.3 * math.exp(-600.0 * diameter), 0.05)


def rain_extinction_marshall_palmer(precipitation_rate: float,
                                    Q_ext: float = 2.0) -> float:
    """Coeficiente de extinção por chuva β [1/m] — modelo teórico.

    Distribuição de Marshall–Palmer [MP48]:
        N(D) = N0·e^(−Λ·D),  N0 = 8e6 m⁻⁴,  Λ = 4100·R^(−0,21) m⁻¹
        (R em mm/h — conversão interna a partir do SI m/s)
    Extinção:  β = (π/4)·Q_ext·∫D²N(D)dD = (π/4)·Q_ext·N0·Γ(3)/Λ³
                 = (π/2)·Q_ext·N0/Λ³
    Q_ext = 2 é exato a <1 % para x = πD/λ ≳ 10³ (paradoxo da extinção,
    van de Hulst 1957) — caso de qualquer gota de chuva em λ óptico; para
    validar com Mie, use `water_refractive_index` + miepython e Q_EXT
    (não Q_scat: a água ABSORVE no NIR, ver [HQ73]).

    Verificação cruzada (forma empírica [Carb98] A=1,076·R^0,67 dB/km):
        R=12,5 mm/h → MP: 7,8 dB/km | Carbonneau: 5,9 dB/km  (razão 1,3;
        MP tende a superestimar levemente a extinção óptica).
    O código original (monodispersa + Stokes) subestimava ~18×.
    """
    if precipitation_rate <= 0:
        return 0.0
    R_mmh = precipitation_rate * 3.6e6          # m/s → mm/h (conversão SI)
    N0 = 8.0e6                                   # m⁻⁴  [MP48]
    Lam = 4100.0 * R_mmh ** (-0.21)              # m⁻¹  [MP48]
    return (math.pi / 2.0) * Q_ext * N0 / Lam ** 3


def rain_attenuation_carbonneau(precipitation_rate: float) -> float:
    """β [1/m] pela forma empírica de Carbonneau [Carb98] (alternativa).

    A[dB/km] = 1,076·R^0,67 (R em mm/h);  β = A/(4,343·10³).
    """
    if precipitation_rate <= 0:
        return 0.0
    R_mmh = precipitation_rate * 3.6e6
    return 1.076 * R_mmh ** 0.67 / (10.0 * math.log10(math.e)) / 1000.0


# ===========================================================================
# 4. Nevoeiro/aerossóis (Kim) e ruído de fase — interfaces em SI
# ===========================================================================
def fog_extinction_kim(visibility: float, wavelength: float) -> float:
    """Coeficiente de extinção por nevoeiro/névoa β [1/m], modelo de Kim.

    Args (SI): visibility [m], wavelength [m].
    Internamente V→km e λ→nm, como publicado em [Kim01].
    """
    V_km = visibility * 1e-3                     # m → km (conversão SI)
    lam_nm = wavelength * 1e9                    # m → nm (conversão SI)
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
    """√(variância de fase por metro) [rad/√m], [AP05] Eq.(75) p.289.

    Args (SI): wavelength [m], height_ag [m].
    Só para uso SEM AtmosphericPhaseProcess (senão conta a turbulência 2×).
    """
    k = 2.0 * math.pi / wavelength
    K0 = 2.0 * math.pi / outer_scale(height_ag)
    return math.sqrt(0.78 * C_n2 * k ** 2 * K0 ** (-5.0 / 3.0))


# ===========================================================================
# 5. Perda total do canal FSO
# ===========================================================================
def channel_FSO_loss(distance: float, wavelength: float, visibility: float,
                     receiver_radius: float, pressure: float,
                     temperature: float, w_0: float, R_0: float,
                     friction_velocity: float, height_ag: float,
                     precipitation_rate: float = 0.0,
                     C_n2: Optional[float] = None,
                     C_T2: Optional[float] = None,
                     Q_ext_rain: float = 2.0,
                     rain_model: str = "marshall_palmer") -> float:
    """Fração de perda do canal FSO ∈ [0,1] (perda MÉDIA de longo prazo).

    Args:
        distance            [m]     comprimento do enlace
        wavelength          [m]     (ex.: 780e-9)
        visibility          [m]     visibilidade meteorológica
        receiver_radius     [m]     raio da abertura do receptor
        pressure            [Pa]    Pressão atmosférica
        temperature         [K]     Temperatura do ar
        w_0                 [m]     cintura inicial do feixe gaussiano
        R_0                 [m]     raio de curvatura inicial (math.inf p/ colimado)
        friction_velocity   [m/s]   u* (correção [C5] — era cm/s ambíguo)
        height_ag           [m]     altura do enlace ACIMA DO SOLO
        precipitation_rate  [m/s]   (1 mm/h = 2,78e-7 m/s)
        C_n2                [m^(-2/3)] (se None, derivado de C_T2 [AP05])
        rain_model          "marshall_palmer" (teórico) | "carbonneau" (empírico)

    Estrutura (como no original, [1] Choudhury & Nandi 2024; [GP22]):
        η_total = η_fog · η_rain · η_turb,  perda = 1 − η_total
        η_turb = 1 − exp(−2a²/w_LT²)  (coleta de longo prazo; a cintilação
        instantânea deve ser tratada por um processo estocástico à parte).
    """
    k = 2.0 * math.pi / wavelength
    l0 = inner_scale(temperature, pressure, friction_velocity, height_ag)

    # --- nevoeiro/aerossóis (Kim) --------------------------------------
    eta_fog = math.exp(-fog_extinction_kim(visibility, wavelength) * distance)

    # --- turbulência: alargamento de longo prazo [GP22, AP05] ----------
    if C_n2 is None:
        if C_T2 is None:
            raise ValueError("Forneça C_n2 ou C_T2.")
        # [AP05]/Murty: Cn = 77,6e-6·(P[mbar]/T²)·(1+7,53e-3/λ_µm²)·C_T
        P_mbar = pressure * 1e-2                 # Pa → mbar (conversão SI)
        lam_um = wavelength * 1e6                # m → µm  (conversão SI)
        C_n2 = ((77.6e-6 * P_mbar / temperature ** 2) ** 2
                * (1.0 + 0.00753 / lam_um ** 2) ** 2 * C_T2)

    Z_R = math.pi * w_0 ** 2 / wavelength                       # Rayleigh
    sigma_R2 = 1.23 * C_n2 * k ** (7.0 / 6.0) * distance ** (11.0 / 6.0)
    w_z2 = w_0 ** 2 * ((1.0 - distance / R_0) ** 2 + (distance / Z_R) ** 2)

    # CORREÇÃO: critério de regime via comprimento de coerência de
    # onda plana ρ0 = (1,46·Cn²·k²·L)^(-3/5) comparado a l0 [AP05, GP22].
    rho0 = (1.46 * C_n2 * k ** 2 * distance) ** (-3.0 / 5.0)
    Lambda_par = 2.0 * distance / (k * w_z2)
    if rho0 < l0:      # moderado-forte (célula mínima l0 domina)
        w_lt2 = w_z2 * (1.0 + 0.74 * (4.0 / 3.0) * sigma_R2
                        * ((35.05 * distance / (k * l0 ** 2)) ** (1.0 / 6.0))
                        * Lambda_par)
    else:              # fraco-moderado
        w_lt2 = w_z2 * (1.0 + 1.63 * sigma_R2 ** (6.0 / 5.0) * Lambda_par)
    eta_turb = 1.0 - math.exp(-2.0 * receiver_radius ** 2 / w_lt2)

    # --- chuva ---------------------------------------------------------
    if rain_model == "marshall_palmer":
        beta_rain = rain_extinction_marshall_palmer(precipitation_rate, Q_ext_rain)
    elif rain_model == "carbonneau":
        beta_rain = rain_attenuation_carbonneau(precipitation_rate)
    else:
        raise ValueError(f"rain_model desconhecido: {rain_model!r}")
    eta_rain = math.exp(-beta_rain * distance)

    return 1.0 - eta_fog * eta_rain * eta_turb


# ===========================================================================
# 6. Processo de fase atmosférica
# ===========================================================================
class AtmosphericPhaseProcess:
    """Fase de pistão φ(t) temporalmente correlacionada ([AP05] Eqs. 75,
    108, 110) — interface convertida para SI (wavelength em METROS).
    Síntese espectral idêntica à original do fork."""

    def __init__(self, duration_s: float, dt_s: float, wavelength: float,
                 C_n2: float, distance: float, outer_scale_m: float,
                 wind_speed_perp: float, seed: Optional[int] = None) -> None:
        if not (0 < dt_s < duration_s):
            raise ValueError("Exige 0 < dt_s < duration_s.")
        self.duration_s = float(duration_s)
        self.dt_s = float(dt_s)
        self.C_n2 = float(C_n2)
        self.distance = float(distance)
        self._k = 2.0 * math.pi / float(wavelength)          # λ em m (SI)
        self._kappa_0 = 2.0 * math.pi / float(outer_scale_m)
        self.wind_speed_perp = float(wind_speed_perp)
        self.tau_atm = 1.0 / (self._kappa_0 * self.wind_speed_perp)
        self.theoretical_variance = (0.78 * self.C_n2 * self._k ** 2
                                     * self.distance
                                     * self._kappa_0 ** (-5.0 / 3.0))
        self._values, self._N = self._synthesize(seed)
        self._clamp_warned = False

    def _psd_one_sided(self, omega: np.ndarray) -> np.ndarray:
        num = (5.82 * self.C_n2 * self._k ** 2 * self.distance
               * self.wind_speed_perp ** (5.0 / 3.0))
        den = (omega ** 2
               + (self._kappa_0 * self.wind_speed_perp) ** 2) ** (4.0 / 3.0)
        return num / den

    def _synthesize(self, seed):
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
        """φ no instante de simulação time_ps (ps — unidade do kernel)."""
        t_s = float(time_ps) * 1e-12
        if t_s <= 0.0:
            return float(self._values[0])
        if t_s >= (self._N - 1) * self.dt_s:
            if not self._clamp_warned:
                warnings.warn("AtmosphericPhaseProcess: consulta além da "
                              "janela pré-gerada; valor saturado.",
                              RuntimeWarning)
                self._clamp_warned = True
            return float(self._values[-1])
        idx = t_s / self.dt_s
        i = int(idx)
        f = idx - i
        return float((1 - f) * self._values[i] + f * self._values[i + 1])


def make_atmospheric_phase_process(distance, timeline_stop_time_ps, ls_params,
                                   loss_parameters, seed=None):
    """Fábrica (interface em SI: ls_params['wavelength'] em m,
    loss_parameters['height_ag'] em m)."""
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

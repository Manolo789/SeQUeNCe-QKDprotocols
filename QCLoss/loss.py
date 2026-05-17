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

"""
from typing import Optional
import miepython
import math
import cmath
from scipy.special import erf
from scipy.integrate import quad
import numpy as np

_C_LIGHT = 2.99792458e8

def outer_scale(friction_velocity: float, height: float):
    '''
    Calculation of the external scale parameter in Kolmogorov turbulence theory

    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 61

    Attributes:
        friction_velocity: Velocidade de atrito [cm/s]
        height: Altura acima do solo [cm]
    '''
    # L_0 is proportional to ε**(1/2)
    # ε: taxa de dissipação de energia turbulenta. A partir da velocidade do vento e da altura, 
    #    usando a teoria da camada limite atmosférica: ε ≈ velocidade_de_atrito³/(κ*h),
    #    onde κ é a constante de von Kármán (κ ≈ 0.4) e h é a altura acima do solo.
    return ((friction_velocity**3)/(0.4*height))**(1/2)
    
def inner_scale(temperature: float, pressure: float, friction_velocity: float, height: float, viscosity: float = None):
    '''
    Calculation of the internal scale parameter in Kolmogorov turbulence theory

    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 57-82

    Attributes:
        temperature: Temperatura ao longo do canal [Kelvin]
        pressure: Pressão atmosférica [milibar]
        friction_velocity: Velocidade de atrito [cm/s]
        height: Altura acima do solo [cm]
        viscosity: Viscosidade do ar [(g/cm)s]
    '''
    # Cálculo da Viscosidade Dinâmica: Equação de Sutherland (mu = mu_0*((T_0+S)/(T+S))*(T/T_0)**(3/2)) 
    # mu_0 = 1.716*1e-5 Kg * m^-1 * s^-1
    # T_0 = 273.15 K
    # S = 110.4 K
    if viscosity == None:
        viscosity = 1.716*1e-4*((273.15+110.4)/(temperature+110.4))*(temperature/273.15)**(3/2)

    # l0_parameter = C_Tatarskii*(ν³/ε)**(1/4)    (referência 7)
    # C_Tatarskii: A constante de Tatarskii relaciona l0_parameter à microescala de Kolmogorov
    # ν: viscosidade cinemática (ν = viscosidade_dinâmica/densidade_do_ar)
    # ε: taxa de dissipação de energia turbulenta. A partir da velocidade do vento e da altura, 
    #    usando a teoria da camada limite atmosférica: ε ≈ velocidade_de_atrito³/(κ*h),
    #    onde κ é a constante de von Kármán (κ ≈ 0.4) e h é a altura acima do solo.
    C_Tatarskii = 7.4
    M_air = 28.9645 # g/mol
    R_ideal = 0.0820574587 # L * atm * K^-1 * mol^-1
    air_density = (M_air*(pressure/1013.25))/(1000*R_ideal*temperature)
    return 0.01*C_Tatarskii*(((viscosity/air_density)**3)/((friction_velocity**3)/(0.4*height)))**(1/4) 

def phase_noise(wavelength: float, C_n2: float, friction_velocity: float, height: float):
    '''
    Calculation of the phase noise coefficient 
     considering the effects of atmospheric turbulence.
    
    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 289

    Attributes:
        wavelength: Comprimento de onda [nm]
        C_n2: Constante de estrutura do índice de refração
        friction_velocity: Velocidade de atrito [cm/s]
        height: Altura acima do solo [cm]
    '''
    # Unit conversions (cm -> m and nm -> m) to keep formulas SI-consistent.
    wavelength_m = wavelength * 1e-9
    k = 2.0 * math.pi / wavelength_m
    K0 = (2*math.pi)/outer_scale(friction_velocity, height)
    
    return math.sqrt(0.78 * C_n2 * (k**2) * (K0**(-5/3)))
    
    
def wind_speed_perp(height: float, ground_wind: float, slew_rate: float = 0):
    """Transverse wind speed V(h) [m/s] from the Bufton wind model.
    
    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. Specifically:
      -Eq. (3), Chapter 12, p. 481:
        V(h) = v_s · h + V_g + 30 · exp[-((h - 9400)/4800)^2]
    
    Attributes:
        height:Altitude h [m]. For horizontal links, use the link 
                altitude above ground (height_link in your case).
        ground_wind: Ground wind speed V_g [m/s]. Default 2 m/s 
                      (light breeze; should be measured locally for production simulations).
        slew_rate: Slew rate v_s [rad/s] for satellite-relative motion. 
                    Default 0 (stationary link).
    """
    tropo_jet = 30.0 * math.exp(-((height - 9400.0) / 4800.0)**2)
    return slew_rate * height + ground_wind + tropo_jet

class AtmosphericPhaseProcess:
    """Pre-generated realization of the temporally correlated atmospheric
    piston phase phi(t), for use as channel noise in a quantum channel.
 
    The process is generated once at construction time over [0, duration]
    with sampling interval dt and sampled by linear interpolation at any
    later request via :meth:`sample`.
    
    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. Specifically:
      - Eq. (75), p. 289: plane-wave phase variance under geometrical
        optics. Experimentally validated -- reference [24] of [1].
      - Eq. (108), p. 296: temporal phase covariance under Taylor
        frozen-flow,
        B_S(tau,L) = sigma^2_S * (kappa_0 V_perp tau)^(5/6) K_{5/6}(kappa_0 V_perp tau).
      - Eq. (110), p. 297: corresponding one-sided power spectrum,
        S(omega) = 5.82 Cn^2 k^2 L V_perp^(5/3) / (omega^2 + (kappa_0 V_perp)^2)^(4/3).

 
    Attributes:
        duration_s: total simulated time span [s].
        dt_s: sampling interval of the underlying array [s].
        wavelength_nm: optical wavelength [nm].
        C_n2: refractive-index structure constant [m^(-2/3)].
        distance: propagation distance [m].
        outer_scale: turbulence outer scale L_0 [m].
        wind_speed_perp: transverse wind speed V_perp [m/s].
        theoretical_variance: Eq. (75) variance to which the sample variance is renormalized [rad^2].
        tau_atm: characteristic correlation time 1/(kappa_0 V_perp) [s].
 
    Notes:
        * Generation cost: O(N log N) at construction, where
            N = duration_s / dt_s.
        * Memory: O(N) floats.
        * If the simulator queries phi(t) with t > duration_s, the value
            at the last sample is returned (clamped); a warning is issued
            once.  This should not occur in correctly configured
            simulations -- choose duration_s >= timeline.stop_time.
    """
 
    def __init__(self, duration_s: float, dt_s: float, wavelength_nm: float, C_n2: float, distance: float, outer_scale: float, wind_speed_perp: float, seed: Optional[int] = None) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive.")
        if dt_s <= 0 or dt_s >= duration_s:
            raise ValueError("dt_s must satisfy 0 < dt_s < duration_s.")
        if outer_scale <= 0:
            raise ValueError("outer_scale must be positive.")
        if wind_speed_perp <= 0:
            raise ValueError("wind_speed_perp must be positive.")
        if C_n2 < 0:
            raise ValueError("C_n2 must be non-negative.")
        if wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive.")
        if distance < 0:
            raise ValueError("distance must be non-negative.")
 
        self.duration_s     = float(duration_s)
        self.dt_s           = float(dt_s)
        self.wavelength_nm  = float(wavelength_nm)
        self.C_n2           = float(C_n2)
        self.distance       = float(distance)
        self.outer_scale    = float(outer_scale)
        self.wind_speed_perp = float(wind_speed_perp)
 
        # Derived physical quantities.
        wavelength_m  = self.wavelength_nm * 1e-9
        self._k       = 2.0 * math.pi / wavelength_m
        self._kappa_0 = 2.0 * math.pi / self.outer_scale
        self.tau_atm  = 1.0 / (self._kappa_0 * self.wind_speed_perp)
 
        # Eq. (75): phase variance under geometrical optics.
        self.theoretical_variance = (0.78 * self.C_n2 * (self._k ** 2) * self.distance * (self._kappa_0 ** (-5.0 / 3.0)))
        # Generate the time series.
        self._values, self._sample_count = self._synthesize(seed)
        self._clamp_warned = False
 
    # ----------------------------------------------------------------
    # Spectral synthesis
    # ----------------------------------------------------------------
    def _psd_one_sided(self, omega: np.ndarray) -> np.ndarray:
        """One-sided temporal phase PSD, [1] Eq. (110)."""
        num = (5.82 * self.C_n2 * (self._k ** 2) * self.distance * (self.wind_speed_perp ** (5.0 / 3.0)))
        den = (omega ** 2 + (self._kappa_0 * self.wind_speed_perp) ** 2) ** (4.0 / 3.0)
        return num / den
 
    def _synthesize(self, seed: Optional[int]) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(seed)
 
        # Total samples: ceil(duration/dt), rounded up to even for rfft cleanliness.
        N = int(math.ceil(self.duration_s / self.dt_s))
        if N % 2:
            N += 1
 
        # Frequency axis of rfft (one-sided, non-negative).
        freqs   = np.fft.rfftfreq(N, d=self.dt_s)
        omegas  = 2.0 * math.pi * freqs
        S       = self._psd_one_sided(omegas)
 
        # Spectral amplitudes: |X[k]|^2 expectation = N*S/(2 dt) for interior k,
        # doubled at DC and Nyquist (which contribute once each in the
        # rfft-Parseval sum).  See derivation in the module docstring.
        amp = np.sqrt(N * S / (2.0 * self.dt_s))
        amp[0] *= math.sqrt(2.0)
        if N % 2 == 0:
            amp[-1] *= math.sqrt(2.0)
 
        # Complex Gaussian noise, with real DC and Nyquist components.
        n_freqs = len(omegas)
        z_re = rng.standard_normal(n_freqs)
        z_im = rng.standard_normal(n_freqs)
        z_im[0] = 0.0
        if N % 2 == 0:
            z_im[-1] = 0.0
 
        X = amp * (z_re + 1j * z_im) / math.sqrt(2.0)
        x = np.fft.irfft(X, n=N)
 
        # Renormalize variance to the experimentally validated Eq. (75).
        # This corrects the ~30 % constant inconsistency between
        # Eq. (75) and the analytic integral of Eq. (110); the shape of
        # the autocovariance [Eq. (108)] is preserved by the rescaling.
        empirical_var = float(np.var(x))
        if empirical_var > 0.0 and self.theoretical_variance > 0.0:
            x *= math.sqrt(self.theoretical_variance / empirical_var)
 
        return x, N
 
    # ----------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------
    def sample(self, time_ps: float) -> float:
        """Return phi at simulation time `time_ps` (in picoseconds).
 
        Linear interpolation between adjacent samples.  Out-of-range
        queries (negative or beyond `duration_s`) are clamped to the
        nearest endpoint with a one-shot warning.
        """
        t_s = float(time_ps) * 1e-12
        if t_s <= 0.0:
            return float(self._values[0])
 
        if t_s >= self._sample_count * self.dt_s - self.dt_s:
            if not self._clamp_warned:
                import warnings
                warnings.warn(
                    f"AtmosphericPhaseProcess queried at t={t_s:.3f} s, "
                    f"beyond pre-generated window {self.duration_s:.3f} s; "
                    "value clamped to endpoint.  Increase `duration_s`.",
                    RuntimeWarning,
                )
                self._clamp_warned = True
            return float(self._values[-1])
 
        # Linear interpolation between neighbouring samples.
        idx_f = t_s / self.dt_s
        i     = int(idx_f)
        frac  = idx_f - i
        return float((1.0 - frac) * self._values[i] + frac * self._values[i + 1])
 
 
# ===========================================================================
#  Factory tied to the simulator's data structures
# ===========================================================================
def make_atmospheric_phase_process(distance: float, timeline_stop_time_ps: float, ls_params: dict, loss_parameters: dict, seed: Optional[int] = None) -> Optional[AtmosphericPhaseProcess]:
    """Build an `AtmosphericPhaseProcess` for one channel.
 
    Returns None if `loss_parameters` lacks the keys necessary to model
    atmospheric phase coherence (in which case the simulator should fall
    back to the legacy Wiener model, or to zero).
 
    Required keys in `loss_parameters`:
        - "C_n2"             : C_n^2 in m^(-2/3)
        - "wind_speed_perp"  : V_perp in m/s
 
    Optional keys:
        - "phase_dt_s"  : sampling interval for the synthesis [s].
                          Default: 0.02 * tau_atm (resolves the corner
                          frequency of the PSD), capped at 1 ms.
    """
 
    L_0     = outer_scale(loss_parameters["friction_velocity"], loss_parameters["height"])
    V_perp  = loss_parameters["wind_speed_perp"]
 
    # Characteristic correlation time tau_atm = L_0 / (2 pi V_perp).
    tau_atm = L_0 / (2.0 * math.pi * V_perp)
    dt_default = max(min(0.02 * tau_atm, 1.0e-3), 1.0e-6)
    dt_s = float(loss_parameters.get("phase_dt_s", dt_default))
 
    # Convert timeline stop time from ps to s.
    duration_s = timeline_stop_time_ps * 1e-12
 
    return AtmosphericPhaseProcess(
        duration_s     = duration_s,
        dt_s           = dt_s,
        wavelength_nm  = ls_params["wavelength"],
        C_n2           = loss_parameters["C_n2"],
        distance       = distance,
        outer_scale    = L_0,
        wind_speed_perp = V_perp,
        seed           = seed,
    )

def polarization_fidelity(distance: float, wavelength: float, w_0: float, receiver_radius: float, photon_number: float, 
    pulse_duration: float, detection_time: float, friction_velocity: float, height: float, C_n2: float, rho_eval: float = None):
    '''
    Calculation of polarization fidelity (degree of polarization of a 
     Gaussian pulse beam) considering atmospheric turbulence effects.

    Based on the results of the article:
    [1] WANG, X. et al. Effects of atmospheric turbulence on the degree 
     of polarization of Gaussian pulse quantum beam. Optik, v. 124, 
     n. 13, p. 1512–1515, jul. 2013.
     
    [2] NIELSEN, M. A.; CHUANG, I. L. Quantum Computation and Quantum
     Information. Cambridge University Press, 2000. §9.3 (relation between
     degree of polarization and fidelity of the depolarizing channel:
     F = sqrt((1 + P) / 2).
    
    [3] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 61

    Attributes:
        distance: Largura do canal [m]
        wavelength: Comprimento de onda [nm]
        w_0: Raio inicial do feixe gaussiano (característica do emissor) [cm]
        receiver_radius: Raio da abertura do receptor [cm]
        photon_number: Número médio de fótons na duração do pulso n_0.
                        Default 1 (single-photon source). Para fontes WCP,
                        use o `mean_photon_num` de `ls_params`.
        pulse_duration: Duração do pulso na fonte T_0 [s]. 
                         Para fontes CW, use T_0 = 1 / frequency.
        detection_time: Janela temporal de detecção T [s]. Para detector com tempo morto
                         fixo, use 1 / count_rate.
        friction_velocity: Velocidade de atrito [cm/s]
        height: Altura acima do solo [cm]
        C_n2: Constante de estrutura do índice de refração
               [m^(-2/3)] -- normalmente já calculada com a função
               `cn2(...)` do seu loss.py.
        rho_eval: Coordenada transversal de avaliação rho [m].
                   Default = receiver_radius / 2 (compromisso entre o
                   eixo e a borda da abertura). Use rho_eval = 0 para
                   a fidelidade no eixo (limite superior).
    '''
    # Unit conversions (cm -> m and nm -> m) to keep formulas SI-consistent.
    wavelength_m = wavelength * 1e-9
    w_0_m = w_0 * 1e-2
    receiver_radius_m = receiver_radius * 1e-2
    k = 2.0 * math.pi / wavelength_m
    L_0 = outer_scale(friction_velocity, height)
    
    # -- rho_0^2 (Below Eq. 8 of [1]) ---
    # C_n^2 is constant throughout the integration interval due to the fact that it is a horizontal ground-to-ground link.
    rho_0_sq = (1/((1.45*(3/8)*distance*C_n2*k**2)**(6/5)))*(1/(1-0.715*((2*math.pi)/L_0)**(1/3)))
    if rho_eval is None:
        rho_eval = 2*receiver_radius_m
    
    # -- T_1 (Below Eq. 10 of [1]) ---
    # Horizontal link with constant C_n^2 -> INTEGRAL = C_n^2 * z.
    # Consider the approximation that the link is horizontal and without 
    #  variation in the height of the nodes. Therefore, C_n^2 remains 
    #  constant (and INTEGRAL_0^z C_n^2(z') dz = C_n^2 * z), since the altitude did not vary.
    T_1 = math.sqrt((pulse_duration**2) + (26.31/(_C_LIGHT**2)) * (L_0**(5.0/3.0)) * C_n2 * distance)
    # -- c parameter (real). Eq. on p.2 of [1]:
    c = ((1/w_0_m**2) + (1/rho_0_sq))**2 + (k**2 / (4*distance**2)) - (1/rho_0_sq**2)
    # -- a parameter (complex). Eq. on p.2 of [1]:
    a = ((1/w_0_m**2) - (1j*k/(2*distance)))**2 - c**2
    # -- b parameter (complex). Eq. on p.2 of [1]:
    b = ((1/w_0_m**2) + (1/rho_0_sq) - (1j*k/(2*distance)))
    # --- d coefficient. Below Eq. (14) of [1]:
    d = (math.pi * photon_number * pulse_duration * math.sqrt(math.pi) / (math.sqrt(2.0)*w_0_m**2)) \ 
        * ((k/(2*math.pi*distance))**2) * erf(math.sqrt(2.0)*(detection_time - distance / _C_LIGHT) / T_1)
    
    # Eq. (14) of [1]
    exponent = -(a * (k ** 2) * (rho_eval ** 2)) / (4.0 * b * c * (distance ** 2))
    exp_term = cmath.exp(exponent)
    ratio = (d * exp_term)/(2.0 * c + (d * exp_term))
    degree_p = max(0.0, min(1.0, abs(cmath.sqrt(ratio))))
    
    # -- Reference [2]
    return math.sqrt((1+degree_p)/2)

def cn2(time: float, sunset: float, sunrise: float, temperature: float, wind_speed: float, rms_wind_speed: float, relative_humidity: float, height: float):
    '''
    Calculation of the refractive index structure constant.
    Based on the results of the article:
    [1] BENDERSKY, S.; KOPEIKA, N. S.; BLAUNSTEIN, N. Atmospheric optical turbulence 
     over land in middle east coastal environments: prediction modeling and measurements. 
     Applied Optics, v. 43, n. 20, p. 4070, 9 jul. 2004.

    [2] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 481

    Attributes:
        time: Hora atual (Exemplo: 15h42m, então time = 15.7) [h]
        sunset: Pôr do sol [h]
        sunrise: Nascer do sol [h]
        temperature: Temperatura do ar [Kelvin]
        wind_speed: Velocidade do Vento [m/s]
        relative_humidity: Umidade Relativa [%]
        height: Altitude [m]

    '''
    th = 12*(time - sunrise)/(sunset - sunrise)
    
    if th <= -4:
        w = 0.11
    elif -4 < th <= -3:
        w = 0.11
    elif -3 < th <= -2:
        w = 0.07
    elif -2 < th <= -1:
        w = 0.08
    elif -1 < th <= 0:
        w = 0.06
    elif 0 < th <= 1:
        w = 0.05
    elif 1 < th <= 2:
        w = 0.1
    elif 2 < th <= 3:
        w = 0.51
    elif 3 < th <= 4:
        w = 0.75
    elif 4 < th <= 5:
        w = 0.95
    elif 5 < th <= 6:
        w = 1.0
    elif 6 < th <= 7:
        w = 0.90
    elif 7 < th <= 8:
        w = 0.80
    elif 8 < th <= 9:
        w = 0.59
    elif 9 < th <= 10:
        w = 0.32
    elif 10 < th <= 11:
        w = 0.22
    elif 11 < th <= 12:
        w = 0.10
    elif 12 < th <= 13:
        w = 0.08
    elif 13 < th:
        w = 0.13
    # 9ºC <= temperature <= 35ºC, 0 <= wind_speed <= 10 m/s, 14% <= relative_humidity <= 92%
    if (282.15 <= temperature <= 308,15) and (0 <= wind_speed <= 10) and (0.14 <= relative_humidity <= 0.92):
        forT = temperature*2*1e-15
        forU = -wind_speed*2.5*1e-15 + (wind_speed**2)*1.2*1e-15 - (wind_speed**3)*8.5*1e-17
        forRH = -(relative_humidity)*2.8*1e-15 + (relative_humidity**2)*2.9*1e-17 - (relative_humidity**3)*1.1*1e-19
        c0 = (w*3.8*1e-14 + forT + forU + forRH -5.3*1e-13)/(15**(-4/3))
        return (5.96e-3)*((rms_wind_speed/27)**2)*((h*1e-5)**10)*math.exp(-height/1000) + (2.7e-16)*math.exp(-height/1500) + c0*math.exp(-height/100)
        

def n_value(wavelength: float, temperature: float, salinity: int = 0):
    '''
    Calculation of the refractive index of a raindrop.
    Based on the results of the article:
    
    Xiaohong Quan and Edward S. Fry, "Empirical equation for the index of 
     refraction of seawater," Appl. Opt. 34, 3477-3480 (1995) 
    '''
    T_C = temperature_K - 273.15 
    lam = wavelength_nm
    S = salinity
    n = 1.31405 + (1.779e-4 - 1.05e-6*T_C + 1.6e-8*T_C**2)*S - 2.02e6*T_C + (15.868 + 0.01155*S - 0.00423*T_C)/lem - 4382/lem**2 + 1.1455e6/lem**3
    return s


def channel_FSO_loss(distance: float, wavelength: float, v_range: float,
                     receiver_radius: float, pressure: float, temperature: float, w_0: float, R_0: float, friction_velocity: float, height: float,
                     size_raindrop: float, viscosity: float, precipitation_rate: float, Q_scat: float, C_n2: float = None, C_T2: float = None, density: float = 1.0, gravitation: float = 980.0):
    '''
    The implementation of the attenuation system in the channel was 
    carried out following the model proposed in the following references:
        [1] DEBARPITA PAUL CHOUDHURY; NANDI, D. Prediction of transmittance for a free space 
        quantum channel and improving quantum Keyrate in adverse atmospheric condition. Optical 
        and quantum electronics, v. 56, n. 6, 3 maio 2024.
            
        [2] MASOUD GHALAII; STEFANO PIRANDOLA. Quantum communications in a moderate-to-
        strong turbulent space. Communications Physics, v. 5, n. 1, 10 fev. 2022. 
            
        [3] FADHIL, H. A. et al. Optimization of free space optics parameters: An optimum solution for bad 
        weather conditions. v. 124, n. 19, p. 3969–3973, 1 out. 2013. 
            
        [4] Ali, M.A.A.: FSO communication characteristics under fog weather condition. Int. J. Sci. Eng. Res. 6(1), 1350–1358 (2015)

        [5] MURTY, S. S. R. Laser beam propagation in atmospheric turbulence. Proceedings of the Indian Academy of 
        Sciences Section C: Engineering Sciences, v. 2, n. 2, p. 179–195, maio 1979.

        [6] PRAHL, S. miepython: Pure python calculation of Mie scattering. Zenodo, 2026. 
        Available at: https://doi.org/10.5281/zenodo.18893972

        [7] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
        SPIE-International Society for Optical Engineering, 2005. p. 57–82

    Attributes:
        distance: Distância [m]
        v_range: Faixa de visibilidade (Visibilidade Horizontal) [Km]
        wavelength: Comprimento de onda [nm]
        
        receiver_radius: Raio de abertura do receptor (característica do receptor) [cm]
        pressure: Pressão atmosférica [milibar]
        temperature: Temperatura ao longo do canal [Kelvin]
        w_0: Raio inicial do feixe gaussiano (característica do emissor) [cm]
        C_T2: Constante de estrutura de temperatura
        C_n2: Constante de estrutura do índice de refração
        R_0: Raio de curvatura inicial da frente de onda do feixe gaussiano (para feixes colimados, adota-se R_0 = math.inf)
        air_density: Densidade do ar [g/cm³]
        friction_velocity: Velocidade de atrito [cm/s]
        height: Altura acima do solo [cm]
        l0_parameter: Medida das distâncias mínimas ao longo das quais as flutuações no índice de refração estão correlacionadas [m]
        
        size_raindrop: Raio da gota de chuva [cm]
        viscosity: Viscosidade do ar [(g/cm)s]
        precipitation_rate: Taxa e precipitação [cm/s]
        Q_scat: Eficiência de dispersão
        density = 1: Densidade da água [g/cm³]
        gravitation = 980: Aceleração da gravidade [cm/s²]
    '''
    wavelength_m = wavelength * 1e-9 # nm to m
    l0_parameter = inner_scale(temperature, pressure, friction_velocity, height)

    # Fog Attenuation (referência 1)
    # Using Kim's model for the dispersion parameter (referência 4)
    if v_range > 50:
        delta = 1.6
    elif 50 > v_range > 6:
        delta = 1.3
    elif 6 > v_range > 1:
        delta = 0.34 + 0.16*v_range
    elif 1 > v_range > 0.5:
        delta = v_range - 0.5
    elif v_range < 0.5:
        delta = 0
    else:
        delta = None # v_range is outside the allowed range or has inconsistent values.
    beta_fog = (3.91/v_range)*((wavelength/550)**(-delta))
    eta_fog = math.exp(-distance*(beta_fog)*1e-3)
    
    # Atmospheric turbulence (referência 1 e 2)
    if C_n2 is None:
        C_n2 = (((77.6*1e-6*pressure)/(temperature**2))**2)*((1+((0.00753)/((wavelength/1000)**2)))**2)*C_T2 # Parâmetro do índice de refração (referência 5)
    k_wave = 2*math.pi/wavelength_m # Número de onda
    Z_R = (math.pi*(w_0*0.01)**2)/wavelength_m # Comprimento do feixe de Rayleigh
    A_rytov = 1.23*(k_wave**(7/6))*C_n2*(distance**(11/6)) # Parâmetro de Rytov
    w_z2 = ((w_0*0.01)**2)*((1-(distance/(R_0*0.01)))**2 + (distance/Z_R)**2)
    zi_parameter = 1/(C_n2*(k_wave**2)*(l0_parameter**(5/3)))
    # Effective beam waist for
    if distance >= zi_parameter:
        w_lt2 = w_z2*(1+0.74*(4/3)*A_rytov*(((35.05*distance)/(k_wave*l0_parameter**2))**(1/6))*((2*distance)/(k_wave*w_z2)))
    elif distance < zi_parameter:
        w_lt2 = w_z2*(1+1.63*(A_rytov**(6/5))*((2*distance)/(k_wave*w_z2)))
    eta_turb = 1 - math.exp(-(2*(receiver_radius*0.01)**2)/(w_lt2))

    # Rain attenuation (referência 1, 3 e 6)
    if Q_scat == None:
        m_water = n_value(wavelength, temperature) + 0j
        Q_scat, _, _, _ = miepython.efficiencies_mx(m_water, (2*math.pi*size_raindrop/wavelength*1e-7))
    limit_s_precipitation = (2*(size_raindrop**2)*density*gravitation)/(9*viscosity)# Velocidade limite de precipitação
    concentration_raindrop = precipitation_rate/((4/3)*math.pi*(size_raindrop**3)*limit_s_precipitation) # Concentração da gotícula de chuva (Distribuição da gota da chuva)
    beta_rain = (math.pi*(size_raindrop**3)*concentration_raindrop*Q_scat/(wavelength*1e-7))
    eta_rain = math.exp(-beta_rain*distance*1e2)
        
    return 1 - (eta_fog*eta_rain*eta_turb)

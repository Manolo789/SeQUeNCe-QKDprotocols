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
from scipy.special import gamma, gammaincc
from scipy.sparse import diags, identity, kron
import numpy as np

_C_LIGHT = 2.99792458e8



def outer_scale(height: float):
    '''
    Calculation of the external scale parameter in Kolmogorov turbulence theory

    Based on the results of the article:
    [1] ANDREWS, L. C.; PHILLIPS, R. L. Laser Beam Propagation Through Random Media.
     SPIE-International Society for Optical Engineering, 2005. p. 483
     
    [2] LUKIN, V. P. Outer scale of atmospheric turbulence. SPIE Proceedings, 
     v. 5981, p. 598101, 6 out. 2005.

    Attributes:
        height: Altura acima do solo [cm]
    '''

    height_m = height * 1e-2
    
    if height_m <= 1:
        return 0.4
    elif 1 < height_m <= 25:
        return 0.4*height_m
    elif 25 < height_m <= 1000:
        return 2*math.sqrt(height_m)
    elif 1000 < height_m:
        return 2*math.sqrt(1000)
    
def inner_scale(temperature: float, pressure: float, friction_velocity: float, height: float, viscosity: float):
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
    K0 = (2*math.pi)/outer_scale(height)
    
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
 
    L_0     = outer_scale(loss_parameters["height"])
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


# ===========================================================================
#  Factory tied to the simulator's data structures
# ===========================================================================
def polarization_jones_vector(polarization):
    """
    Converte a especificação de polarização em um vetor de Jones
    normalizado (eps1, eps2), com o modo 1 = horizontal (a1) e o modo
    2 = vertical (a2) dos operadores de Stokes.

    Formatos aceitos:
      a) rótulo (str): 'H','V','D+45','D-45','R','L' (e sinônimos);
      b) vetor de Jones explícito (ex, ey) complexos (eliptica geral);
      c) parâmetros da elipse (theta, delta) reais [rad].
    O vetor retornado já vem normalizado (|eps1|^2 + |eps2|^2 = 1).
    """
    if isinstance(polarization, str):
        p = polarization.strip().upper()
        table = {
            'H': (1, 0), 'HORIZONTAL': (1, 0),
            'V': (0, 1), 'VERTICAL': (0, 1),
            'D': (1, 1), 'D+45': (1, 1), '45': (1, 1),
            'A': (1, -1), 'D-45': (1, -1), '135': (1, -1),
            'R': (1, -1j), 'RCP': (1, -1j), 'RIGHT': (1, -1j),
            'L': (1, 1j), 'LCP': (1, 1j), 'LEFT': (1, 1j),
        }
        if p not in table:
            raise ValueError(f"Polarização desconhecida: {polarization!r}")
        ex, ey = table[p]

    elif isinstance(polarization, (tuple, list)) and len(polarization) == 2:
        a, b = polarization
        if isinstance(a, complex) or isinstance(b, complex):
            # vetor de Jones já fornecido explicitamente
            ex, ey = a, b
        else:
            # (theta, delta) -> elipse de polarização geral
            theta, delta = a, b
            ex = math.cos(theta)
            ey = math.sin(theta) * cmath.exp(1j * delta)
    else:
        raise ValueError(
            "polarization deve ser um rótulo str, um vetor de Jones "
            "(ex, ey) ou parâmetros de elipse (theta, delta)."
        )

    ex, ey = complex(ex), complex(ey)
    norm = math.sqrt(abs(ex) ** 2 + abs(ey) ** 2)
    if norm == 0:
        raise ValueError("Vetor de polarização nulo.")
    return ex / norm, ey / norm

def two_mode_ladder_operators(N):
    """
    Constrói os operadores de aniquilação a1, a2 de dois modos (modo 1 =
    horizontal, modo 2 = vertical), truncados em N fótons por modo, na
    base de Fock |n1> (x) |n2>.

    Operador de aniquilação de modo único: a|n> = sqrt(n) |n-1>.
    Operadores de dois modos por produto de Kronecker:
        a1 = a (x) I ,   a2 = I (x) a .
    Matrizes esparsas (csr) de dimensão (N+1)^2 x (N+1)^2.
    """
    # a|n> = sqrt(n)|n-1>  ->  entradas sqrt(1..N) na 1a super-diagonal
    a = diags(np.sqrt(np.arange(1, N + 1)), offsets=1, format="csr")
    Id = identity(N + 1, format="csr")
    a1 = kron(a, Id, format="csr")   # aniquila no modo horizontal
    a2 = kron(Id, a, format="csr")   # aniquila no modo vertical
    return a1, a2


def polarized_fock_state(eps, n0):
    """
    Vetor de estado |n0, eps> = (b^dagger)^{n0} / sqrt(n0!) |0,0>,
    com b^dagger = eps1 a1^dagger + eps2 a2^dagger, construído aplicando
    n0 vezes o operador de criação do modo de polarização ao vácuo.

    Retorna um numpy array denso de tamanho (n0+1)^2.
    """
    eps1, eps2 = eps
    N = n0                                   # N = n0 fotons/modo bastam
    a1, a2 = two_mode_ladder_operators(N)
    b_dag = eps1 * a1.getH() + eps2 * a2.getH()   # operador de criacao do modo b

    dim = (N + 1) ** 2
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0                              # vacuo |0,0>  (indice 0)
    for _ in range(n0):                       # aplica b^dagger n0 vezes
        psi = b_dag.dot(psi)
    norm = np.linalg.norm(psi)
    if norm > 0:                              # normaliza (== dividir por sqrt(n0!))
        psi = psi / norm
    return psi

def _coherence_matrix(eps, n0):
    """
    Matriz de coerência da fonte 2x2

        N0[i-1, j-1] = < n0, eps | a^dagger_i a_j | n0, eps > ,

    para i, j em {1, 2}. Retorna uma tupla ((N0_11, N0_12),(N0_21, N0_22))
    """
    if n0 == 0:
        return ((0j, 0j), (0j, 0j))
    N = n0
    a1, a2 = two_mode_ladder_operators(N)
    a = {1: a1, 2: a2}
    psi = polarized_fock_state(eps, n0)

    N0 = [[0j, 0j], [0j, 0j]]
    for i in (1, 2):
        for j in (1, 2):
            op = a[i].getH().dot(a[j])           # a^dagger_i a_j (esparsa)
            N0[i - 1][j - 1] = complex(np.vdot(psi, op.dot(psi)))
    return ((N0[0][0], N0[0][1]), (N0[1][0], N0[1][1]))

def initial_number_operator(i, j, polarization, n0):
    """
    Elemento (i, j) da matriz de coerência da fonte,
    n0_ij = < a^dagger_i a_j >, obtido a partir dos operadores de
    criação/aniquilação dos fótons.
 
      - n0 inteiro (>= 0): usa a construção quântica explícita -- monta o
        estado de Fock polarizado |n0, eps> aplicando b^dagger ao vácuo e
        avalia < a^dagger_i a_j > sobre ele.
 
      - n0 não inteiro (>= 0): o estado de Fock não está definido; utiliza-se
        a forma fechada do primeiro momento
 
              n0 * conj(eps_i) * eps_j ,
 
        que é o resultado exato para um estado coerente (Glauber) de média
        n0 = |alpha|^2, típico de pulsos coerentes fracos (WCP). Veja a observação
        
    Observação: fontes com número médio de fótons não inteiro (coerente / WCP)
    A construção de Fock exige n0 inteiro (número de fótons definido ou apenas
    um fóton para o caso SPS). Para uma fonte em estado coerente (Glauber)
    polarizada ao longo de eps, com amplitude alpha e número médio 
    n0 = |alpha|^2 real (não necessariamente inteiro) -- caso tipico 
    dos pulsos coerentes fracos (WCP) em QKD -- o mesmo cálculo do 
    primeiro momento fornece
 
        < a^dagger_i a_j > = n0 * conj(eps_i) * eps_j ,
 
    identica em forma ao caso de Fock. Isto é, a matriz de coerência (primeiro
    momento) não distingue um estado de Fock de um estado coerente de mesma
    média n0; por isso initial_number_operator() tambem aceita n0 não inteiro,
    usando diretamente essa forma fechada.
 
    A diferença entre as duas estatísticas aparece apenas no segundo momento do
    número total, que entra no grau de polarização via < S0(S0+2) >:
    - Fock (número definido):   S0 e nítido (var = 0)  -> < S0(S0+2) > = n0(n0+2);
    - coerente (Poissoniano):   var(S0) = n0           -> < S0(S0+2) > = n0(n0+3).
    """

    if n0 < 0:
        raise ValueError(f"n0 deve ser não-negativo (recebido n0={n0!r}).")
    eps = polarization_jones_vector(polarization)
    if float(n0).is_integer():
        # caso SPS: valor esperado sobre o estado de Fock polarizado
        N0 = _coherence_matrix(eps, int(n0))
        return N0[i - 1][j - 1]
    # fonte coerente / WCP: primeiro momento em forma fechada
    return n0 * eps[i - 1].conjugate() * eps[j - 1]

def n(i, j, loss_percent, polarization, n0):
    """
    Elemento (i, j) da matriz de coerencia APOS o canal,

        n_ij = tau * < a^dagger_i a_j > ,   tau = 1 - loss_percent,

    onde < a^dagger_i a_j > e agora o valor esperado quantico do operador
    numero inicial sobre o estado de n0 fotons polarizados (Secao 3),
    NAO mais um produto externo postulado.

    Consistencia com o artigo (Eqs. 12/19): para polarization='H' e
    n0 = n11, resulta n_11 = tau*n11 e n_22 = n_12 = n_21 = 0.
    """
    tau = 1 - loss_percent
    return tau * initial_number_operator(i, j, polarization, n0)

def aa_function(i, j, k, w0, z, rho0, rho, q, loss_percent, polarization, n0):
    pi = math.pi
    A_term = (((1/w0**2) + (1/rho0**2))**2+(k**2 / (4 * z**2)))*rho0**4 - 1
    B_term = (1/w0**2) - ((1j*k)/(2*z)) + (1/rho0**2)
    exp_term = (-((A_term) + ((B_term)*rho0**2 -1)**2)/(4*(B_term)*(A_term)))*(((k*rho)/z) + q)**2
    n_ij = n(i, j, loss_percent, polarization, n0)
    return (((2*pi*n_ij)/w0**2)*((k/(2*pi*z))**2) / (((1/w0**2) + (1/rho0**2))**2 + (k**2 / (4 * z**2)) - (1/rho0**4)))*cmath.exp(exp_term)

def polarization_fidelity(distance: float, wavelength: float, w_0: float, 
    L0: float, l0: float, C_n2: float, alpha: float, loss_percent: float, 
    polarization="H", n0: float = 1.0, rho: float = 0):
    '''
    Calculation of polarization fidelity (degree of polarization 
     for single-foton beam) considering atmospheric turbulence effects.

    Based on the results of the article:
    [1] WANG, Y. et al. Degree of polarization for quantum light field
        propagating through non-Kolmogorov turbulence. Opt. Laser Technol.
        43, 776-780 (2011). (Eqs. 1, 5, 12, 19)
    [2] E. Collett, Stokes parameters for quantum systems, Am. J. Phys. 38,
        563-574 (1970).
    [3] M. Chekhova, P. Banzer, Polarization of Light: In Classical, Quantum,
        and Nonlinear Optics, De Gruyter (2021), Cap. 11.
    [4] R. J. Glauber, Coherent and incoherent states of the radiation field,
        Phys. Rev. 131, 2766-2788 (1963). (estados coerentes -- "de Glauber")


    Attributes:
        distance: Largura do canal [m]
        wavelength: Comprimento de onda [nm]
        w_0: Raio inicial do feixe gaussiano (característica do emissor) [cm]
        L0 : escala externa da turbulencia [m]
        l0 : escala interna da turbulencia [m]
        C_n2: Constante de estrutura do índice de refração
               [m^(-2/3)] -- normalmente já calculada com a função
               `cn2(...)` do seu loss.py.
        alpha : constante fractal (expoente) da turbulencia nao-Kolmogorov
        polarization : Polarização de entrada -- ver
                       polarization_jones_vector() (rotulo 'H'/'V'/'D+45'/'D-45'/
                       'R'/'L', vetor de Jones (ex,ey) ou parametros de elipse
                       (theta, delta))
        n0 : número médio de fotons do feixe na fonte (default 1, fonte
             de foton unico)
        loss_percent : perdas adicionais do canal (0 a 1)
        rho: Coordenada transversal de avaliação rho [m]. (default 0, no eixo,
              limite superior do grau de polarizacao).
                   
    Return:
        P: grau de polarização (numero real entre 0 e 1)
    '''
    wavelength_m = wavelength * 1e-9
    w_0_m = w_0 * 1e-2
    k = 2.0 * math.pi / wavelength_m
    z = distance
    q = (6.62607015*1e-34) / wavelength_m # E = h*v -> p = h/λ 
    
    A_function = (1/(4*(math.pi**2)))*gamma(alpha - 1)*cos((alpha*math.pi)/2)
    c_function = (gamma(5-(alpha/2))*A_function*(2*math.pi)/3)**(1/(alpha - 5))
    k0 = (2*math.pi)/L0
    k_m = c_function/l0
    beta = 2*k0**2 - 2*k_m**2 + alpha*k_m**2
    rho0 = (((((math.pi*k)**2)*z*A_function*C_n2)/(6*(alpha-2)))*((k_m**(2-alpha))*beta*cmath.exp((k0/k_m)**2)*gammaincc(2-(alpha/2), (k0/k_m)**2) - 2*k0**(4-alpha)))**(-1/2)
    
    common = dict(k=k, w0=w_0_m, z=z, rho0=rho0, rho=rho, q=q, loss_percent=loss_percent, polarization=polarization, n0=n0)
    S0 = aa_function(1, 1, **common) + aa_function(2, 2, **common)
    S1 = aa_function(1, 1, **common) - aa_function(2, 2, **common)
    S2 = aa_function(1, 2, **common) + aa_function(2, 1, **common)
    S3 = 1j*(aa_function(2, 1, **common) - aa_function(1, 2, **common))
    P_squared = (S1**2 + S2**2 + S3**2) / (S0*(S0 + 2))

    return cmath.sqrt(P_squared).real

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
        rms_wind_speed: Velocidade do Vento na troposfera [m/s]
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
    # ATTENTION: This model uses the $C_n^2$ function from reference [1] as a FIRST APPROXIMATION. Proper operation is 
    # not guaranteed for values ​​outside the following range: 9ºC <= temperature <= 35ºC, 0 <= wind_speed <= 10 m/s, 
    # 14% <= relative_humidity <= 92%. In the future, it will be necessary to structure a model consistent with 
    # the simulated geographic region. The development of this model can be tracked at <github.com/...>.
    # 

    forT = temperature*2*1e-15
    forU = -wind_speed*2.5*1e-15 + (wind_speed**2)*1.2*1e-15 - (wind_speed**3)*8.5*1e-17
    forRH = -(relative_humidity)*2.8*1e-15 + (relative_humidity**2)*2.9*1e-17 - (relative_humidity**3)*1.1*1e-19
    c0 = (w*3.8*1e-14 + forT + forU + forRH -5.3*1e-13)/(15**(-4/3))
    return (5.96e-3)*((rms_wind_speed/27)**2)*((height*1e-5)**10)*math.exp(-height/1000) + (2.7e-16)*math.exp(-height/1500) + c0*math.exp(-height/100)

        

def n_value(wavelength: float, temperature: float, salinity: int = 0):
    '''
    Calculation of the refractive index of a raindrop.
    Based on the results of the article:
    
    Xiaohong Quan and Edward S. Fry, "Empirical equation for the index of 
     refraction of seawater," Appl. Opt. 34, 3477-3480 (1995) 
    '''
    T_C = temperature - 273.15 
    lam = wavelength
    S = salinity
    n = 1.31405 + (1.779e-4 - 1.05e-6*T_C + 1.6e-8*T_C**2)*S - 2.02e-6*T_C + (15.868 + 0.01155*S - 0.00423*T_C)/lam - 4382/lam**2 + 1.1455e6/lam**3
    return n


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
        precipitation_rate: Taxa de precipitação [cm/s]
        Q_scat: Eficiência de dispersão
        density = 1: Densidade da água [g/cm³]
        gravitation = 980: Aceleração da gravidade [cm/s²]
        
    Returns:
        float: loss rate for transmitted photons (in %/100)
    '''
    wavelength_m = wavelength * 1e-9 # nm to m
    # Cálculo da Viscosidade Dinâmica: Equação de Sutherland (mu = mu_0*((T_0+S)/(T+S))*(T/T_0)**(3/2)) 
    # mu_0 = 1.716*1e-5 Kg * m^-1 * s^-1
    # T_0 = 273.15 K
    # S = 110.4 K
    if viscosity == None:
        viscosity = 1.716*1e-4*((273.15+110.4)/(temperature+110.4))*(temperature/273.15)**(3/2)
    l0_parameter = inner_scale(temperature, pressure, friction_velocity, height, viscosity)

    # Fog Attenuation (referência 1)
    # Using Kim's model for the dispersion parameter (referência 4)
    if v_range > 50:
        delta = 1.6
    elif 50 >= v_range > 6:
        delta = 1.3
    elif 6 >= v_range > 1:
        delta = 0.34 + 0.16*v_range
    elif 1 >= v_range > 0.5:
        delta = v_range - 0.5
    elif v_range <= 0.5:
        delta = 0
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
    if Q_scat is None:
        m_water = n_value(wavelength, temperature) + 0j
        Q_scat, _, _, _ = miepython.efficiencies_mx(m_water, (2*math.pi*size_raindrop/wavelength*1e-7))
    limit_s_precipitation = (2*(size_raindrop**2)*density*gravitation)/(9*viscosity)# Velocidade limite de precipitação
    concentration_raindrop = precipitation_rate/((4/3)*math.pi*(size_raindrop**3)*limit_s_precipitation) # Concentração da gotícula de chuva (Distribuição da gota da chuva)
    beta_rain = (math.pi*(size_raindrop**3)*concentration_raindrop*Q_scat/(wavelength*1e-7))
    eta_rain = math.exp(-beta_rain*distance*1e2)
        
    return 1 - (eta_fog*eta_rain*eta_turb)

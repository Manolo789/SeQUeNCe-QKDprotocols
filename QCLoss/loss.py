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
import math

def channel_FSO_loss(distance: float, wavelength: float, v_range: float,
                     receiver_radius: float, pressure: float, temperature: float, w_0: float, C_T: float, R_0: float,
                     size_raindrop: float, viscosity: float, precipitation_rate: float, Q_scat: float, density: float = 1.0, gravitation: float = 980.0):
    '''
    The implementation of the attenuation system in the channel was 
    carried out following the model proposed in the following references:
        -DEBARPITA PAUL CHOUDHURY; NANDI, D. Prediction of transmittance for a free space 
        quantum channel and improving quantum Keyrate in adverse atmospheric condition. Optical 
        and quantum electronics, v. 56, n. 6, 3 maio 2024.
            
        -MASOUD GHALAII; STEFANO PIRANDOLA. Quantum communications in a moderate-to-
        strong turbulent space. Communications Physics, v. 5, n. 1, 10 fev. 2022. 
            
        -FADHIL, H. A. et al. Optimization of free space optics parameters: An optimum solution for bad 
        weather conditions. v. 124, n. 19, p. 3969–3973, 1 out. 2013. 
            
        -Ali, M.A.A.: FSO communication characteristics under fog weather condition. Int. J. Sci. Eng. Res. 6(1), 1350–1358 (2015)
    Attributes:
        distance: Distância [m]
        v_range: Faixa de visibilidade (Visibilidade Horizontal) [Km]
        wavelength: Comprimento de onda [nm]
        
        receiver_radius: Raio de abertura do receptor (característica do receptor) [cm]
        pressure: Pressão atmosférica [milibar]
        temperature: Temperatura ao longo do canal [Kelvin]
        w_0: Raio inicial do feixe gaussiano (característica do emissor) [cm]
        C_T: Constante de estrutura de temperatura
        R_0: Raio de curvatura inicial da frente de onda do feixe gaussiano (para feixes colimados, adota-se R_0 = math.inf)
        
        size_raindrop: Raio da gota de chuva [cm]
        viscosity: Viscosidade do ar [(g/cm)s]
        precipitation_rate: Taxa e precipitação [cm/s]
        Q_scat: Eficiência de dispersão
        density = 1: Densidade da água [g/cm³]
        gravitation = 980: Aceleração da gravidade [cm/s²]
    '''
    wavelength_m = wavelength * 1e-9 # nm to m
    # Fog Attenuation
    
    # Using Kim's model for the dispersion parameter
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

    beta_fog = (3.92/v_range)*((wavelength/550)**(-delta))
    eta_fog = math.exp(-distance*(beta_fog)*1e-3)
    
    # Atmospheric turbulence
    C_n2 = (((77.6*1e-6*pressure)/(temperature**2))**2)*((1+((0.00753)/((wavelength/1000)**2)))**2)*C_T**2 # Parâmetro do índice de refração
    k_wave = 2*math.pi/wavelength_m # Número de onda
    Z_R = (math.pi*(w_0*0.01)**2)/wavelength_m # Comprimento do feixe de Rayleigh
    A_rytov = 1.23*(k_wave**(7/6))*C_n2*(distance**(11/6)) # Parâmetro de Rytov
    w_z2 = ((w_0*0.01)**2)*((1-(distance/R_0))**2 + (distance/Z_R)**2)
    w_lt2 = w_z2*(1+1.63*A_rytov*((2*distance)/(k_wave*w_z2)))**2 # Effective beam waist
    eta_turb = 1 - math.exp(-(2*(receiver_radius*0.01)**2)/(w_lt2))

    # Rain attenuation    
    limit_s_precipitation = (2*(size_raindrop**2)*density*gravitation)/(9*viscosity)# Velocidade limite de precipitação
    concentration_raindrop = precipitation_rate/((4/3)*math.pi*(size_raindrop**3)*limit_s_precipitation) # Concentração da gotícula de chuva (Distribuição da gota da chuva)
    beta_rain = (math.pi*(size_raindrop**2)*concentration_raindrop*Q_scat)
    eta_rain = math.exp(-beta_rain*distance*1e2)
        
    return 1 - (eta_fog*eta_rain*eta_turb)

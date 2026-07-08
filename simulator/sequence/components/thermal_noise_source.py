"""
======================================================================
Thermal background photon source for the SeQUeNCe simulator -- License
======================================================================

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

======================================================================

"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization
from ..utils.encoding_cow import time_bin_cow
from QCLoss.sky_radiance import n_background

_h = 6.62607015e-34   # J·s
_c = 2.99792458e8     # m/s


class ThermalNoiseSource(Entity):
    """Source of background thermal photons for FSO link.

    Emite fótons com estado aleatório, modelando a radiância
    do céu que penetra pela abertura do receptor.

    Attributes:
        n_B (float)      : fótons de fundo por modo [adimensional]
        frequency (float): frequência de clock [Hz]
        encoding_type (dict): tipo de codificação
        active (bool)    : liga/desliga a fonte
    """

    def __init__(self, name: str, timeline: "Timeline", n_B: float, frequency: float, encoding_type: dict = None) -> None:
        Entity.__init__(self, name, timeline)

        if encoding_type is None:
            encoding_type = polarization

        self.n_B = n_B
        self.frequency = frequency
        self.encoding_type = encoding_type
        self.active = True

        # Taxa total de chegada de fótons de fundo [fótons/s]
        self._arrival_rate = n_B * frequency

    def init(self) -> None:
        """Agenda o primeiro evento de emissão de fóton de fundo."""
        if self._arrival_rate > 0 and self.active:
            self._schedule_next()

    def _schedule_next(self) -> None:
        """Agenda próximo fóton de fundo"""
        # Intervalo exponencial: E[T] = 1/λ  (em segundos)
        dt_s = self.get_generator().exponential(1.0 / self._arrival_rate)
        dt_ps = int(dt_s * 1e12)

        t_next = self.timeline.now() + max(dt_ps, 1)
        process = Process(self, "_emit", [])
        event = Event(t_next, process)
        self.timeline.schedule(event)

    def _emit(self) -> None:
        """Cria um fóton de fundo com estado aleatório e o envia ao detector."""
        if not self.active:
            return

        state  = self._random_state()
        photon = Photon(
            name          = "bg_photon",
            timeline      = self.timeline,
            encoding_type = self.encoding_type,
            quantum_state = state,
        )
        # Fóton real → passa pela eficiência η_eff do Detector
        self._receivers[0].get(photon)

        # Agenda o próximo fóton
        self._schedule_next()

    def _random_state(self) -> tuple:
        """Estado quântico aleatório para simular fundo incoerente.

        Polarização: estado aleatório na esfera de Bloch (equatorial)
        time_bin_cow: early ou late com probabilidade igual
        """
        rng = self.get_generator()

        if self.encoding_type["name"] == "polarization":
            # Ângulo de polarização uniforme em [0, π)
            theta = rng.uniform(0, math.pi)
            return (complex(math.cos(theta)), complex(math.sin(theta)))

        elif self.encoding_type["name"] in ("time_bin", "time_bin_cow"):
            # Chegada no bin early (0) ou late (1) com p=0.5
            if rng.random() < 0.5:
                return time_bin_cow["early"] if "cow" in self.encoding_type["name"] \
                       else (complex(1), complex(0))
            else:
                return time_bin_cow["late"] if "cow" in self.encoding_type["name"] \
                       else (complex(0), complex(1))
        else:
            # Fallback genérico: superposição uniforme
            return (complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)))

    def set_n_B(self, n_B: float) -> None:
        """Atualiza n_B em tempo de execução (ex: transição noite→dia)."""
        self.n_B = n_B
        self._arrival_rate = n_B * self.frequency

    #def update_from_params(self, wavelength_nm: float, delta_lambda_nm: float, delta_t_ns: float, omega_fov_sr: float, a_R_cm: float, B_sky: float) -> float:
    #    """Calcula n_B via Eq.(32) e atualiza a fonte. Retorna n_B calculado."""
    #    n_B = n_background(wavelength_nm, delta_lambda_nm, delta_t_ns, omega_fov_sr, a_R_cm, B_sky)
    #    self.set_n_B(n_B)
    #    return n_B
    

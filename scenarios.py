"""Diurnal atmospheric profile and scenario materialiser for QKD sweeps.

Provides a simple time-of-day model that maps an hour in [0, 24) onto a
self-consistent set of atmospheric parameters (T, P, U, RH, ...) and feeds
them into the loss-parameter and thermal-noise dicts consumed by
:func:`QKD_Extension.sim_scenario`.
"""

import math
from QCLoss.loss import cn2, wind_speed_perp, f_velocity, viscosity_shuterland
import pandas as pd
from datetime import datetime

def diurnal_profile(hour, *, base_loss_parameters, base_thermal_params,
                ls_params, sunrise, sunset, link_altitude_m, date, dataframe):
    """Build (loss_parameters, thermal_params, ls_overrides) for one hour.

    The return shape (3-tuple) is the exact contract expected by
    :func:`QKD_Extension.sim_scenario` for a `materialize_fn`. The third
    element (``ls_overrides``) is ``None`` because the laser is not
    affected by the diurnal cycle in this model; if a more advanced
    scenario needs to tweak the light source (e.g. emission frequency)
    it should return a dict here.

    Args:
        hour (float): local time in fractional hours (0..23). Example: hour=13,515 (13:30:54)
        base_loss_parameters (dict): base loss-parameter dict. Only the
            atmospheric quantities are overwritten by the hourly profile;
            everything else (geometry, source aperture, raindrop size,
            etc.) is preserved verbatim.
        base_thermal_params (dict): base thermal-noise dict. Only
            ``B_sky`` is overwritten (day/night switch).
        ls_params (dict): light-source params. Currently unused -- kept
            in the signature for symmetry with :func:`sim_scenario` and
            forward-compatibility with scenarios that vary the laser.
        sunrise, sunset (float): local solar times in fractional hours.
        link_altitude_m (float): link mid-altitude above sea level [m].

    Returns:
        tuple: (loss_parameters, thermal_params, ls_overrides).
    """
    '''
    horas = int(hour)
    resto_minutos = (hour - horas) * 60
    minutos = int(resto_minutos)
    segundos = round((resto_minutos - minutos) * 60)

    # Ajuste caso o arredondamento dos segundos ou minutos chegue a 60
    if segundos == 60:
        segundos = 0
        minutos += 1
    if minutos == 60:
        minutos = 0
        horas += 1
    
    date = datetime.strptime(date+" "+str(horas)+":"+str(minutos)+":"+str(segundos), "%Y-%m-%d %H:%M:%S")
    '''
    date = datetime.fromisoformat(date) + timedelta(hours=hour)
    correspondencias = dataframe.index[dataframe[0] == date]
    index = correspondencias[0] if len(correspondencias) > 0 else None
    
    T = float(dataframe[4][index]) + 273.15
    P = float(dataframe[3][index])
    u = float(dataframe[7][index])
    u_star = f_velocity(wind_speed=u, T_classification=7, height_ag=base_loss_parameters["height"]/100)
    RH = float(dataframe[5][index])
    rms = 21
    p_rate = ((float(dataframe[6][index]))/60)/10  # Passo temporal: 1 minuto. Logo, se há X mm Chuva_Tot em 60 s,
                                                       # então a taxa de precipitação neste instante é X/60 mm/s.
                                                       # ([mm]/60)/10 = [cm/s]
    cn = cn2(time=hour, sunset=sunset, sunrise=sunrise,
             temperature=T,
             wind_speed=u,
             rms_wind_speed=rms,
             relative_humidity=RH,
             height=link_altitude_m)

    lp = {**base_loss_parameters,
          "temperature":        T,
          "pressure":           P,
          "friction_velocity":  u_star,
          "wind_speed_perp":    wind_speed_perp(link_altitude_m, u),
          "precipitation_rate": p_rate,
          "viscosity": viscosity_shuterland(T),
          "C_n2":               cn}

    # B_sky changes by ~4 orders of magnitude between night and day.
    # Values from Pirandola (Phys. Rev. Res. 3, 023130, 2021).
    is_night = (hour < sunrise) or (hour > sunset)
    tp = {**base_thermal_params,
          "B_sky": 1.5e-6 if is_night else 1.5e-2}

    return lp, tp, None

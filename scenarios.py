"""Diurnal atmospheric profile and scenario materialiser for QKD sweeps.

Provides a time-of-day model that maps an hour in [0, 24) onto a
self-consistent set of atmospheric parameters (T, P, U, RH, ...) and feeds
them into the loss-parameter and thermal-noise dicts consumed by
:func:`QKD_Extension.sim_scenario`.
"""

import math
from QCLoss.loss import (f_velocity, viscosity_sutherland, wind_speed_perp, cn2_horizontal_link)
from QCLoss.sky_radiance import b_sky_at
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
    
    T  = float(dataframe[4][index]) + 273.15          # °C -> K
    P  = float(dataframe[3][index]) * 100.0           # hPa -> Pa   (SI!)
    u  = float(dataframe[7][index])                   # m/s
    u_star = f_velocity(u, T_classification=7,
                    height_ag=base_loss_parameters["height_ag"])  # m/s [C5]
    RH = float(dataframe[5][index])
    p_rate = float(dataframe[6][index]) / 60.0 * 1e-3  # mm/min -> m/s (SI)
    
    cn = cn2(time=hour, sunset=sunset, sunrise=sunrise,
             temperature=T,
             wind_speed=u,
             rms_wind_speed=rms,
             relative_humidity=RH,
             height=link_altitude_m)

    lp = {**base_loss_parameters, "temperature": T, "pressure": P,
          "friction_velocity": u_star,
          "wind_speed_perp": wind_speed_perp(site_altitude, u),
          "precipitation_rate": p_rate, "C_n2": cn}

    # B_sky contínuo (substitui o chaveamento binário dia/noite):
    when = (datetime.fromisoformat(date) + timedelta(hours=hour)).replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
    tp = {**base_thermal_params, "B_sky": b_sky_at(when, latitude, longitude, ls_params["wavelength"], pressure=P)}

    return lp, tp, None

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
from datetime import datetime, timedelta, timezone


def diurnal_profile(hour, *, base_loss_parameters, base_thermal_params,
                ls_params, sunrise, sunset, site_altitude, latitude,
                longitude, local_tz, date, dataframe):
    """Build (loss_parameters, thermal_params, ls_overrides) for one hour.

    The return shape (3-tuple) is the exact contract expected by
    :func:`QKD_Extension.sim_scenario` for a `materialize_fn`. The third
    element (``ls_overrides``) is ``None`` because the laser is not
    affected by the diurnal cycle in this model; if a more advanced
    scenario needs to tweak the light source (e.g. emission frequency)
    it should return a dict here.

    Args:
        hour (float): local time in fractional hours (0..23). Example: hour=13.515 (13:30:54)
        base_loss_parameters (dict): base loss-parameter dict. Only the
            atmospheric quantities are overwritten by the hourly profile;
            everything else (geometry, source aperture, raindrop size,
            etc.) is preserved verbatim.
        base_thermal_params (dict): base thermal-noise dict. Only
            ``B_sky`` is overwritten (continuous solar-elevation model).
        ls_params (dict): light-source params. Only ``wavelength`` is read
            (by ``b_sky_at``); kept in the signature for symmetry with
            :func:`sim_scenario` and for scenarios that vary the laser.
        sunrise, sunset (float): local solar times in fractional hours.
        site_altitude (float): site altitude above sea level [m], used by
            ``wind_speed_perp``.
        latitude, longitude (float): site coordinates [rad], used to
            compute the solar elevation.
        local_tz (timezone): time zone of the measurement timestamps, so
            they can be converted to UTC for the solar position.
        date (str): ISO date of the measurement day ("YYYY-MM-DD").
        dataframe (pd.DataFrame): meteorological table indexed by
            timestamp in column 0, with pressure (3), temperature (4),
            relative humidity (5), precipitation (6) and wind speed (7).

    Returns:
        tuple: (loss_parameters, thermal_params, ls_overrides).
    """
    when_local = datetime.fromisoformat(date) + timedelta(hours=hour)
    matches = dataframe.index[dataframe[0] == when_local]
    index = matches[0] if len(matches) > 0 else None
    
    T  = float(dataframe[4][index]) + 273.15          # °C -> K
    P  = float(dataframe[3][index]) * 100.0           # hPa -> Pa   (SI!)
    u  = float(dataframe[7][index])                   # m/s
    u_star = f_velocity(u, T_classification=7,
                    height_ag=base_loss_parameters["height_ag"])  # m/s [C5]
    RH = float(dataframe[5][index])
    p_rate = float(dataframe[6][index]) / 60.0 * 1e-3  # mm/min -> m/s (SI)
    
    cn = cn2_horizontal_link(base_loss_parameters["height_ag"], hour=hour,
                         sunrise=sunrise, sunset=sunset, temperature=T,
                         wind_speed=u, relative_humidity=RH) 

    lp = {**base_loss_parameters, "temperature": T, "pressure": P,
          "friction_velocity": u_star,
          "wind_speed_perp": wind_speed_perp(site_altitude, u),
          "precipitation_rate": p_rate, "C_n2": cn}

    # Continuous B_sky from the solar elevation (replaces the former
    # binary day/night switch).
    when = when_local.replace(tzinfo=local_tz).astimezone(timezone.utc)
    tp = {**base_thermal_params, "B_sky": b_sky_at(when, latitude, longitude, ls_params["wavelength"], pressure=P)}

    return lp, tp, None

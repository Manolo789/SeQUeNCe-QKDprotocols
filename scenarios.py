"""Diurnal atmospheric profile and scenario materialiser for QKD sweeps.

Provides a simple time-of-day model that maps an hour in [0, 24) onto a
self-consistent set of atmospheric parameters (T, P, U, RH, ...) and feeds
them into the loss-parameter and thermal-noise dicts consumed by
:func:`QKD_Extension.sim_scenario`.
"""

import math
from QCLoss.loss import cn2, wind_speed_perp


def diurnal_profile(hour: float) -> dict:
    """Return atmospheric parameters for a given hour of the day.

    Args:
        hour (float): local time in fractional hours (0..24).

    Returns:
        dict with keys:
            - temperature       [K]
            - pressure          [mbar]
            - wind_speed        [m/s]
            - friction_velocity [cm/s]
            - relative_humidity [fraction, 0..1]
            - rms_wind_speed    [m/s]
            - precipitation_rate
            - hour              (echoed back for traceability)

    Notes:
        This is a placeholder analytical model -- replace by measured
        data (e.g. a CSV from a local weather station) for production
        simulations. The shape of the curves (minimum near 6h, maximum
        near 15h, etc.) is intentionally simple and meant for sanity
        checks of the simulator pipeline, not for quantitative claims.
    """
    # Temperatura: mínima ~6h, máxima ~15h
    T = 296.0 + 6.0 * math.sin(2 * math.pi * (hour - 9) / 24)        # K
    # Pressão varia pouco ao longo do dia
    P = 927 + 2 * math.sin(2 * math.pi * hour / 24)                  # mbar
    # Vento mais forte de tarde
    u = 1.5 + 1.0 * math.sin(2 * math.pi * (hour - 13) / 24)         # m/s
    u_star = max(10.0, u * 100 * 0.1)                                # cm/s
    RH = 0.65 - 0.25 * math.sin(2 * math.pi * (hour - 15) / 24)      # fração
    rms = max(10.0, u * 1.5)                                         # m/s

    return dict(temperature=T, pressure=P, wind_speed=u,
                friction_velocity=u_star, relative_humidity=RH,
                rms_wind_speed=rms, precipitation_rate=0.0,
                hour=hour)


def materialize(hour, *, base_loss_parameters, base_thermal_params,
                ls_params, sunrise, sunset, link_altitude_m):
    """Build (loss_parameters, thermal_params, ls_overrides) for one hour.

    The return shape (3-tuple) is the exact contract expected by
    :func:`QKD_Extension.sim_scenario` for a `materialize_fn`. The third
    element (``ls_overrides``) is ``None`` because the laser is not
    affected by the diurnal cycle in this model; if a more advanced
    scenario needs to tweak the light source (e.g. emission frequency)
    it should return a dict here.

    Args:
        hour (float): local time in fractional hours (0..24).
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
    prof = diurnal_profile(hour)

    cn = cn2(time=prof["hour"], sunset=sunset, sunrise=sunrise,
             temperature=prof["temperature"],
             wind_speed=prof["wind_speed"],
             rms_wind_speed=prof["rms_wind_speed"],
             relative_humidity=prof["relative_humidity"],
             height=link_altitude_m)

    lp = {**base_loss_parameters,
          "temperature":        prof["temperature"],
          "pressure":           prof["pressure"],
          "friction_velocity":  prof["friction_velocity"],
          "wind_speed_perp":    wind_speed_perp(link_altitude_m,
                                                prof["wind_speed"]),
          "precipitation_rate": prof["precipitation_rate"],
          "C_n2":               cn}

    # B_sky changes by ~4 orders of magnitude between night and day.
    # Values from Pirandola (Phys. Rev. Res. 3, 023130, 2021).
    is_night = (hour < sunrise) or (hour > sunset)
    tp = {**base_thermal_params,
          "B_sky": 1.5e-6 if is_night else 1.5e-2}

    return lp, tp, None

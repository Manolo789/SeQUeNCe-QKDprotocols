"""Diurnal atmospheric profile and scenario materialiser for QKD sweeps.

Provides a time-of-day model that maps an hour in [0, 24) onto a
self-consistent set of atmospheric parameters (T, P, U, RH, ...) and feeds
them into the loss-parameter and thermal-noise dicts consumed by
:func:`QKD_Extension.sim_scenario`.

TIME CONVENTION (single source of truth, see
:func:`QKD_Extension.default_environment`): ``hour``, ``sunrise`` and
``sunset`` are LOCAL fractional hours of the day ``date`` in the zone
``local_tz``. Every UTC instant needed by the solar model is derived from
them through :func:`local_hour_to_utc`, never hard-coded, so that C_n2 and
B_sky are always evaluated at the SAME instant.
"""

import math
import warnings
from QCLoss.loss import (f_velocity, viscosity_sutherland, wind_speed_perp, cn2_horizontal_link)
from QCLoss.sky_radiance import (b_sky_at, b_sky_from_diffuse,
                                 solar_elevation, B_NIGHT_FULL_MOON)
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Column layout of 'sensores/estação-solar-usp_Tabela01.dat' (Campbell TOA5,
# read with header=None and skiprows=4, hence positional indices).
# ---------------------------------------------------------------------------
COL_TIMESTAMP    = 0    # TIMESTAMP
COL_ERROR        = 2    # Erro
COL_PRESSURE     = 3    # PressaoAr_Avg   [mbar]
COL_TEMPERATURE  = 4    # TempAr_Avg      [deg C]
COL_HUMIDITY     = 5    # UmidRel         [%]
COL_RAIN         = 6    # Chuva_Tot       [mm/min]
COL_WIND         = 7    # VelVento_Avg    [m/s]
COL_RAD_GLOBAL   = 13   # RadGlobal_Avg   [W/m^2]
COL_RAD_DIFFUSE  = 21   # RadDifuso_Avg   [W/m^2]
COL_IRRADIANCE   = 29   # Irradiancia_Avg [W/m^2]


def local_hour_to_utc(date, hour, local_tz):
    """UTC instant of a LOCAL fractional hour of a given day.

    Args:
        date (str): ISO date of the measurement day ("YYYY-MM-DD").
        hour (float): local time in fractional hours (0..24).
        local_tz (timezone): fixed-offset zone of the measurements.

    Returns:
        datetime: the same instant, tz-aware, in UTC.
    """
    naive = datetime.fromisoformat(date) + timedelta(hours=float(hour))
    return naive.replace(tzinfo=local_tz).astimezone(timezone.utc)


def diurnal_profile(hour, *, base_loss_parameters, base_thermal_params,
                ls_params, sunrise, sunset, site_altitude, latitude,
                longitude, local_tz, date, dataframe,
                b_sky_source="measured", spectral_width=None):
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
            ``B_sky`` is overwritten; ``filter_bandwidth`` is read as the
            Delta_lambda of the empirical B_sky.
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
        dataframe (pd.DataFrame): meteorological table of the station,
            indexed by timestamp in column 0; the columns actually read are
            listed in the ``COL_*`` constants of this module (pressure,
            temperature, humidity, rain, wind, global/diffuse radiation and
            the irradiance channel).
        b_sky_source (str): ``"measured"`` (default) computes B_sky from
            the diffuse radiation of the record, via
            :func:`QCLoss.sky_radiance.b_sky_from_diffuse`; ``"model"``
            keeps the theoretical clear-sky chain of
            :func:`QCLoss.sky_radiance.b_sky_at`. Any unusable record
            (missing/negative diffuse) falls back to the model with a
            RuntimeWarning, so the sweep never breaks on a bad sample.
        spectral_width (float | None): Delta_lambda [m] of the empirical
            B_sky; None uses the receiver ``filter_bandwidth``.

    Returns:
        tuple: (loss_parameters, thermal_params, ls_overrides).

    Raises:
        ValueError: if the day/hour has no record in ``dataframe``, or if
            ``b_sky_source`` is not one of "measured"/"model".
    """
    if b_sky_source not in ("measured", "model"):
        raise ValueError("diurnal_profile: b_sky_source must be 'measured' "
                         f"or 'model' (got {b_sky_source!r}).")
    when_local = datetime.fromisoformat(date) + timedelta(hours=float(hour))
    matches = dataframe.index[dataframe[COL_TIMESTAMP] == when_local]
    if len(matches) == 0:
        raise ValueError(
            f"diurnal_profile: no record at {when_local:%Y-%m-%d %H:%M} in "
            "the station table; check the 'date' of the site and whether the "
            "requested hour exists in the file.")
    index = matches[0]

    err = dataframe[COL_ERROR][index]
    if pd.notna(err) and float(err) != 0.0:
        warnings.warn(
            f"diurnal_profile: record {when_local:%Y-%m-%d %H:%M} flagged "
            f"with Erro={float(err):g} by the datalogger; the sample is used "
            "as is.", RuntimeWarning)

    T  = float(dataframe[COL_TEMPERATURE][index]) + 273.15   # °C -> K
    P  = float(dataframe[COL_PRESSURE][index]) * 100.0       # mbar -> Pa (SI!)
    u  = float(dataframe[COL_WIND][index])                   # m/s
    u_star = f_velocity(u, T_classification=7,
                    height_ag=base_loss_parameters["height_ag"])  # m/s [C5]
    RH = float(dataframe[COL_HUMIDITY][index])
    # Chuva_Tot is the TOTAL of the 1 min record: mm/min -> m/s (SI).
    p_rate = float(dataframe[COL_RAIN][index]) / 60.0 * 1e-3
    
    cn = cn2_horizontal_link(base_loss_parameters["height_ag"], hour=hour,
                         sunrise=sunrise, sunset=sunset, temperature=T,
                         wind_speed=u, relative_humidity=RH) 

    lp = {**base_loss_parameters, "temperature": T, "pressure": P,
          "friction_velocity": u_star,
          "wind_speed_perp": wind_speed_perp(site_altitude, u),
          "precipitation_rate": p_rate, "C_n2": cn}

    # SAME instant as C_n2 above: the local hour is converted through
    # `local_tz`, never through a hard-coded UTC timestamp.
    when = local_hour_to_utc(date, hour, local_tz)
    b_model = b_sky_at(when, latitude, longitude, ls_params["wavelength"],
                       pressure=P)

    if b_sky_source == "measured":
        # The pyranometer reads 0 at night, so the floor below the horizon
        # is the theoretical value (twilight/night regimes of b_sky).
        sun_up = solar_elevation(when, latitude, longitude) > 0.0
        try:
            b_value = b_sky_from_diffuse(
                dataframe[COL_RAD_DIFFUSE][index],
                base_thermal_params["filter_bandwidth"],
                spectral_width=spectral_width,
                global_irradiance=dataframe[COL_RAD_GLOBAL][index],
                irradiance=dataframe[COL_IRRADIANCE][index],
                b_night=B_NIGHT_FULL_MOON if sun_up else b_model)
        except ValueError as exc:
            warnings.warn(
                f"diurnal_profile: {exc}; falling back to the theoretical "
                f"B_sky at {when_local:%Y-%m-%d %H:%M}.", RuntimeWarning)
            b_value = b_model
    else:
        b_value = b_model

    tp = {**base_thermal_params, "B_sky": b_value}

    return lp, tp, None

import math
from QCLoss.loss import cn2, wind_speed_perp

def diurnal_profile(hour: float) -> dict:
    """Retorna parâmetros atmosféricos para uma hora do dia.

    Args:
        hour (float): hora local (0..24).
    Returns:
        dict com chaves: temperature, pressure, wind_speed, friction_velocity,
                         relative_humidity, rms_wind_speed, precipitation_rate
    """
    # Modelo simples senoidal — substitua por dados reais (CSV/labmicro)
    # Temperatura: mínima às 6h, máxima às 15h
    T = 296.0 + 6.0 * math.sin(2*math.pi*(hour - 9)/24)   # K
    # Pressão varia pouco
    P = 927 + 2*math.sin(2*math.pi*hour/24)               # mbar
    # Vento mais forte de tarde
    u = 1.5 + 1.0*math.sin(2*math.pi*(hour-13)/24)        # m/s (módulo)
    u_star = max(10.0, u*100*0.1)                         # cm/s, escala rugosidade
    RH = 0.65 - 0.25*math.sin(2*math.pi*(hour-15)/24)     # fração
    rms = max(10.0, u*1.5)                                # m/s
    return dict(temperature=T, pressure=P, wind_speed=u,
                friction_velocity=u_star, relative_humidity=RH,
                rms_wind_speed=rms, precipitation_rate=0.0,
                hour=hour)

def materialize(hour, *, base_loss_params, base_thermal_params,
                ls_params, sunrise, sunset, height_link_cm, link_altitude_m):
    """Constrói (loss_parameters, thermal_params) para um instante do dia."""
    prof = diurnal_profile(hour)

    cn = cn2(time=prof["hour"], sunset=sunset, sunrise=sunrise,
             temperature=prof["temperature"], wind_speed=prof["wind_speed"],
             rms_wind_speed=prof["rms_wind_speed"],
             relative_humidity=prof["relative_humidity"],
             height=link_altitude_m)

    lp = {**base_loss_params,
          "temperature":        prof["temperature"],
          "pressure":           prof["pressure"],
          "friction_velocity":  prof["friction_velocity"],
          "wind_speed_perp":    wind_speed_perp(link_altitude_m,
                                                prof["wind_speed"]),
          "precipitation_rate": prof["precipitation_rate"],
          "C_n2":               cn}

    # thermal_params depende de delta_lambda_nm e do count_rate (fixos), e
    # opcionalmente de B_sky que muda dia/noite — bom exemplo de acoplamento.
    is_night = (hour < sunrise) or (hour > sunset)
    tp = {**base_thermal_params,
          "B_sky": 1.5e-6 if is_night else 1.5e-2}

    return lp, tp

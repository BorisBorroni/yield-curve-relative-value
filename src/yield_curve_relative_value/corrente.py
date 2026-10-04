"""Analisi della curva piu' recente: dove si trovano oggi residui e farfalle rispetto allo storico.

Lo storico (data/storico/) e' fisso; i giorni successivi si scaricano a ogni esecuzione in
data/corrente/ (cartella non versionata) e il calcolo e' rifatto da zero.
"""
import numpy as np
import pandas as pd

from . import analisi as an
from . import dati
from . import strategia as sg

FINESTRA = 60
FARFALLE = {"2-5-10": (2, 5, 10), "5-10-30": (5, 10, 30)}


def anni_da_scaricare(oggi: pd.Timestamp) -> range:
    """Anni da scaricare per coprire i giorni dopo la fine dello storico."""
    return range(pd.Timestamp(dati.FINE_STORICO).year + 1, oggi.year + 1)


def percentile(valore: float, riferimento) -> float:
    """Quota (0-100) dei valori storici assoluti minori o uguali al valore assoluto dato."""
    r = np.abs(np.asarray(riferimento, dtype=float))
    r = r[~np.isnan(r)]
    return float(100 * np.mean(r <= abs(valore)))


def tabella_oggi(par_storico, residui_storico, par_recente, finestra: int = FINESTRA) -> pd.DataFrame:
    """Per ogni scadenza, ultimo giorno disponibile: rendimento, residuo di Nelson-Siegel (bp), z-score
    sui 60 giorni precedenti e percentile del |z| rispetto a tutto lo storico.

    I residui dei giorni recenti si calcolano con lo stesso adattamento giornaliero dello storico."""
    _, res_recenti = an.adatta_serie(par_recente)
    res = pd.concat([residui_storico, res_recenti])
    z = pd.DataFrame(sg.zscore(res.to_numpy(), finestra), index=res.index, columns=res.columns)
    z_storico = z[z.index <= pd.Timestamp(dati.FINE_STORICO)]
    ultimo = res.index[-1]
    righe = []
    for c in res.columns:
        righe.append({"scadenza": c, "rendimento": float(par_recente[c].iloc[-1]),
                      "residuo_bp": round(float(res[c].iloc[-1]), 2), "z": round(float(z[c].iloc[-1]), 2),
                      "percentile_|z|": round(percentile(z[c].iloc[-1], z_storico[c].to_numpy()), 1)})
    t = pd.DataFrame(righe).set_index("scadenza")
    t.attrs["ultimo_giorno"] = ultimo
    return t


def farfalle_oggi(par_storico, par_recente, tau: float = an.TAU_DL, finestra: int = FINESTRA) -> pd.DataFrame:
    """Valore, z-score e percentile del |z| delle farfalle (pesi DV01 e neutrali ai fattori) all'ultimo giorno."""
    par = pd.concat([par_storico, par_recente])
    righe = []
    for nome, terna in FARFALLE.items():
        for neutr in ("dv01", "fattori"):
            x = sg.serie_farfalla(par, terna, sg.pesi_farfalla(terna, neutr, tau)).dropna()
            z = pd.Series(sg.zscore(x.to_numpy(), finestra), index=x.index)
            storico = z[z.index <= pd.Timestamp(dati.FINE_STORICO)]
            righe.append({"farfalla": nome, "neutralita": neutr, "valore_bp": round(float(x.iloc[-1]), 2),
                          "z": round(float(z.iloc[-1]), 2),
                          "percentile_|z|": round(percentile(z.iloc[-1], storico.to_numpy()), 1)})
    return pd.DataFrame(righe).set_index(["farfalla", "neutralita"])

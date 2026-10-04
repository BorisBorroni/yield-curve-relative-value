"""Segnale z-score di ritorno alla media e farfalle (butterfly) sulla curva.

Convenzioni (le stesse del mondo sintetico):
- il segnale usa solo il passato: z_t = (x_t - media)/dev. std sui `finestra` giorni PRIMA di t;
- posizione +1 se z > soglia, -1 se z < -soglia, 0 altrimenti; si scommette sul rientro verso la media;
- guadagno al giorno t+1 = posizione_t * (x_t - x_{t+1}) in punti base, meno il costo;
- costo = costo_bp * somma dei valori assoluti dei pesi * |variazione di posizione|: e' il costo, in bp di
  rendimento, per ogni unita' di DV01 scambiata su ciascuna gamba.
I pesi delle gambe sono in unita' di DV01: un peso +1 e' un titolo comprato che guadagna 1 per ogni bp
di calo del rendimento.
"""
import numpy as np
import pandas as pd

from . import curva as cv


def zscore(x, finestra: int = 60) -> np.ndarray:
    """z_t = (x_t - media)/dev. std sui `finestra` giorni PRIMA di t (il giorno t e' escluso); NaN prima."""
    x = np.asarray(x, dtype=float)
    z = np.full_like(x, np.nan)
    for t in range(finestra, len(x)):
        w = x[t - finestra : t]
        z[t] = (x[t] - w.mean(axis=0)) / w.std(axis=0, ddof=1)
    return z


def posizioni(x, finestra: int = 60, soglia: float = 1.0) -> np.ndarray:
    """Posizione (+1, 0, -1) giorno per giorno per una serie o per le colonne di una tabella."""
    z = zscore(x, finestra)
    pos = np.where(z > soglia, 1.0, np.where(z < -soglia, -1.0, 0.0))
    pos[:finestra] = 0.0
    return pos


def emivita(x, finestra: int) -> np.ndarray:
    """Emivita (giorni) del ritorno alla media stimata con un AR(1) sui `finestra` giorni precedenti;
    infinita se il coefficiente non e' fra 0 e 1 (la serie non torna alla media nella finestra)."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.inf)
    for t in range(finestra, len(x)):
        w = x[t - finestra : t]
        a, b = w[:-1] - w[:-1].mean(), w[1:] - w[1:].mean()
        phi = (a @ b) / (a @ a) if a @ a > 0 else np.nan
        if 0 < phi < 1:
            out[t] = -np.log(2) / np.log(phi)
    return out


def guadagni_serie(segnale_bp, pnl_bp, somma_pesi: float = 1.0, finestra: int = 60, soglia: float = 1.0,
                   costo_bp: float = 0.0, emivita_max: float | None = None):
    """Guadagno giornaliero (bp) di una serie. `segnale_bp` genera la posizione, `pnl_bp` e' la serie su cui
    si guadagna (le due coincidono nei dati reali). Con emivita_max si opera solo se l'emivita stimata nella
    finestra e' al piu' quel numero di giorni. Restituisce (guadagni, posizioni)."""
    pos = posizioni(segnale_bp, finestra, soglia)
    if emivita_max is not None:
        pos = np.where(emivita(segnale_bp, finestra) <= emivita_max, pos, 0.0)
    e = np.asarray(pnl_bp, dtype=float)
    lordo = pos[:-1] * (e[:-1] - e[1:])
    cambi = np.abs(np.diff(pos, axis=0, prepend=0.0))[:-1]
    return lordo - costo_bp * somma_pesi * cambi, pos


def pesi_farfalla(scadenze, neutralita: str, tau: float = 1.37) -> np.ndarray:
    """Pesi (in DV01) di una farfalla corta-pancia-lunga con la pancia a -1.

    - "dv01": ali 0.5 e 0.5, somma dei pesi zero (neutrale al livello);
    - "fattori": ali scelte in modo da annullare l'esposizione al livello e alla pendenza di
      Nelson-Siegel (carichi con il tau indicato): resta solo la curvatura.
    """
    a, b, c = (float(s) for s in scadenze)
    if neutralita == "dv01":
        return np.array([0.5, -1.0, 0.5])
    if neutralita != "fattori":
        raise ValueError("neutralita deve essere 'dv01' o 'fattori'")
    pend = cv.carichi([a, b, c], tau)[:, 1]  # carico della pendenza
    m = np.array([[1.0, 1.0], [pend[0], pend[2]]])
    wa, wc = np.linalg.solve(m, [1.0, pend[1]])
    return np.array([wa, -1.0, wc])


def serie_farfalla(par: pd.DataFrame, scadenze, pesi) -> pd.Series:
    """Rendimento pesato della farfalla in bp: somma di peso_i * rendimento_i. La posizione +1 compra le ali
    e vende la pancia, e guadagna quando questa serie scende."""
    return pd.Series((par[list(scadenze)].to_numpy() @ np.asarray(pesi)) * 100, index=par.index)

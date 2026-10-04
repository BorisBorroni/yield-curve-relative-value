"""Analisi sui dati reali: adattamento giornaliero, premio on-the-run, PCA, previsione.

Tutti i rendimenti sono par yield in %, le differenze in punti base (bp).
Due sottoperiodi: il Tesoro ha cambiato metodo di costruzione della
curva il 6 dicembre 2021 (spline quasi-cubico prima, monotone convex dopo).
"""
import numpy as np
import pandas as pd

from . import curva as cv
from . import statistica as st

CAMBIO_METODO = pd.Timestamp("2021-12-06")
TAU_DL = 1.37  # tau di Diebold-Li: lambda 0.0609 per mese, cioe' 1.37 anni
RITARDI_NW = 250  # circa un anno di borsa: le serie dei premi hanno autocorrelazione alta anche a 60 giorni


def adatta_serie(par: pd.DataFrame, svensson: bool = False):
    """Adatta la curva ogni giorno ai par yield osservati (colonne = scadenze in anni).

    Ogni giorno e' adattato da solo (stima con piu' punti di partenza, vedi adatta_par_yield).
    Restituisce (parametri, residui):
    parametri = tabella beta0..beta3, tau1, tau2 per giorno; residui = osservato - curva, in bp.
    """
    n = par.columns.to_numpy(dtype=float)
    righe, res = [], []
    for giorno, riga in par.iterrows():
        y = riga.to_numpy(dtype=float)
        try:
            p = cv.adatta_par_yield(n, y, svensson=svensson)
        except ValueError as e:
            raise ValueError(f"{giorno.date()}: {e}") from e
        righe.append([p.beta0, p.beta1, p.beta2, p.beta3, p.tau1, p.tau2])
        res.append((y - cv.par_yield_vettore(n, p)) * 100)  # NaN dove il dato manca
    parametri = pd.DataFrame(righe, index=par.index, columns=["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"])
    return parametri, pd.DataFrame(res, index=par.index, columns=par.columns)


def curva_da_parametri(parametri: pd.DataFrame, scadenze) -> pd.DataFrame:
    """Par yield (%) delle curve descritte da una tabella di parametri."""
    n = np.asarray(scadenze, dtype=float)
    righe = [cv.par_yield_vettore(n, cv.Parametri(*r)) for r in parametri[["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]].to_numpy()]
    return pd.DataFrame(righe, index=parametri.index, columns=scadenze)


def differenza_bp(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a - b in bp sui giorni e le scadenze in comune."""
    giorni = a.index.intersection(b.index)
    return (a.loc[giorni] - b.loc[giorni, a.columns]) * 100


def periodi(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"fino a dic 2021": df[df.index < CAMBIO_METODO],
            "da dic 2021": df[df.index >= CAMBIO_METODO],
            "intero": df}


def riepilogo(diff_bp: pd.DataFrame, ritardi: int = RITARDI_NW) -> pd.DataFrame:
    """Per scadenza e periodo: media, mediana, dev. std (bp), t di Newey-West della media, n giorni.

    Le serie sono molto persistenti (autocorrelazione a 60 giorni fra 0.5 e 0.93): il t ordinario sarebbe
    troppo ottimista e anche venti ritardi di Newey-West sono pochi, per questo si usano 250 ritardi
    (prudente: su 1017 giorni, il sottoperiodo piu' corto, tende a sottostimare il t). I p-value non sono corretti per test multipli."""
    righe = []
    for nome, d in periodi(diff_bp).items():
        for c in d.columns:
            x = d[c].dropna()
            if len(x) < 50:
                continue
            m, t = st.newey_west(x.to_numpy(), ritardi)
            righe.append({"periodo": nome, "scadenza": c, "media": m, "mediana": x.median(),
                          "dev_std": x.std(), "t_NW": t, "giorni": len(x)})
    return pd.DataFrame(righe).set_index(["periodo", "scadenza"]).round(2)


def pca(variazioni: pd.DataFrame, componenti: int = 3):
    """PCA sulla matrice di covarianza delle variazioni giornaliere (bp). Restituisce
    (quota di varianza spiegata, punteggi [giorni x componenti], autovettori [scadenze x componenti])."""
    x = variazioni.dropna().to_numpy()
    x = x - x.mean(axis=0)
    val, vet = np.linalg.eigh(np.cov(x, rowvar=False))
    ordine = np.argsort(val)[::-1]
    val, vet = val[ordine], vet[:, ordine]
    punteggi = pd.DataFrame(x @ vet[:, :componenti], index=variazioni.dropna().index)
    return val[:componenti] / val.sum(), punteggi, pd.DataFrame(vet[:, :componenti], index=variazioni.columns)


def r2_su_fattori(punteggi: pd.DataFrame, fattori: pd.DataFrame) -> pd.Series:
    """R2 della regressione di ogni componente principale sulle variazioni dei tre fattori di Nelson-Siegel
    (insieme): misura quanto i tre fattori coprono lo spazio delle componenti, a prescindere dal segno."""
    idx = punteggi.index.intersection(fattori.index)
    x = np.column_stack([np.ones(len(idx)), fattori.loc[idx].to_numpy()])
    r2 = {}
    for c in punteggi.columns:
        y = punteggi.loc[idx, c].to_numpy()
        b, *_ = np.linalg.lstsq(x, y, rcond=None)
        r2[c] = 1 - np.sum((y - x @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    return pd.Series(r2)


def fattori_dl(par: pd.DataFrame, tau: float = TAU_DL) -> pd.DataFrame:
    """Beta di Diebold-Li (regressione lineare con tau fisso, trattando i par yield come rendimenti)."""
    beta, _, _ = cv.residui_panel(par.to_numpy(), par.columns.to_numpy(dtype=float), tau1=tau)
    return pd.DataFrame(beta, index=par.index, columns=["livello", "pendenza", "curvatura"])


def previsione_dl(par: pd.DataFrame, orizzonte: int, inizio_test: str, tau: float = TAU_DL):
    """Previsione fuori campione a `orizzonte` mesi: AR(1) sui beta contro passeggiata casuale.

    Si usa l'ultimo giorno di ogni mese, su un calendario mensile completo: un mese senza dati (come gli anni
    2002-2006, senza i 30 anni) resta vuoto e nessuna coppia (beta_s, beta_{s+h}) lo attraversa. Per ogni
    mese t dell'insieme di test, l'AR(1) diretto beta_{t+h} = a + phi beta_t si stima sulle sole coppie
    disponibili con s + h <= t (finestra espansiva). Passeggiata casuale: beta_{t+h} previsto = beta_t.
    Restituisce (rmse in bp per scadenza dei due metodi, t di Diebold-Mariano con Newey-West
    sulla perdita quadratica media fra le scadenze, numero di previsioni, t di Clark-West).

    L'AR(1) contiene la passeggiata casuale come caso particolare (a = 0, phi = 1): fra modelli annidati il t
    di Diebold-Mariano non e' normale nemmeno asintoticamente, e Clark-West corregge la perdita dell'AR(1)
    per il rumore di stima. Si rifiuta la passeggiata casuale al 5% (una coda) se il t di Clark-West supera 1.645.
    """
    pieno = par.dropna()
    ultimo = pieno.groupby(pieno.index.to_period("M")).tail(1)
    ultimo.index = ultimo.index.to_period("M")
    mensile = ultimo.reindex(pd.period_range(ultimo.index[0], ultimo.index[-1], freq="M"))
    ok = mensile.notna().all(axis=1).to_numpy()
    b = np.full((len(mensile), 3), np.nan)
    b[ok] = fattori_dl(mensile[ok], tau).to_numpy()
    x = cv.carichi(par.columns.to_numpy(dtype=float), tau)
    y = mensile.to_numpy()
    t0 = int(mensile.index.searchsorted(pd.Period(inizio_test, freq="M")))
    err_ar, err_rw = [], []
    for t in range(t0, len(b) - orizzonte):
        if not (ok[t] and ok[t + orizzonte]):
            continue
        prev = np.empty(3)
        for j in range(3):
            xs, ys = b[: t + 1 - orizzonte, j], b[orizzonte : t + 1, j]  # coppie (beta_s, beta_{s+h}) con s+h <= t
            valide = ~np.isnan(xs) & ~np.isnan(ys)
            phi, a = np.polyfit(xs[valide], ys[valide], 1)
            prev[j] = a + phi * b[t, j]
        err_ar.append((y[t + orizzonte] - x @ prev) * 100)
        err_rw.append((y[t + orizzonte] - x @ b[t]) * 100)
    ea, er = np.array(err_ar), np.array(err_rw)
    rmse = pd.DataFrame({"AR(1)": np.sqrt((ea**2).mean(axis=0)), "RW": np.sqrt((er**2).mean(axis=0))}, index=par.columns)
    perdita = (er**2).mean(axis=1) - (ea**2).mean(axis=1)  # > 0 se l'AR(1) sbaglia meno
    ritardi = max(orizzonte, 1) - 1 + 2
    _, dm = st.newey_west(perdita, ritardi)
    clark_west = (er**2 - (ea**2 - (er - ea) ** 2)).mean(axis=1)
    _, cw = st.newey_west(clark_west, ritardi)
    return rmse.round(1), round(dm, 2), len(ea), round(cw, 2)

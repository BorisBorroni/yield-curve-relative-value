"""Mondo sintetico: una curva nota, con rumore e mispricing noti, per sapere cosa il metodo puo' vedere.

La curva vera ha fattori che seguono processi AR(1) giornalieri. A ogni scadenza si somma un
mispricing OU (mean-reverting, con deviazione standard e emivita note) e un rumore di misura.
Tutto in rendimenti (%), con mispricing e rumore dichiarati in punti base.

Il rumore e' ERRORE DI MISURA (non e' un prezzo a cui si puo' operare): disturba il segnale ma
i guadagni si calcolano sui prezzi senza rumore. Se invece si valutassero anche sul rumore, ogni
rumore indipendente darebbe un guadagno "eseguibile" fasullo, perche' rientra da solo.

La strategia di prova opera sul RESIDUO della curva adattata ogni giorno: e' un limite
superiore per una vera strategia a farfalla, che deve anche coprirsi e pagare i costi su
piu' gambe. Se non si vede nemmeno qui, non si vedra' nelle gambe reali.
"""
import numpy as np

from . import curva as cv
from . import statistica as st
from . import strategia

SCADENZE = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
TAU_VERO = 2.0
# media, persistenza giornaliera e innovazione giornaliera dei tre fattori (in %)
FATTORI = {"media": (4.5, -1.5, -1.0), "phi": (0.999, 0.998, 0.995), "sigma": (0.04, 0.05, 0.08)}
# quarto fattore della variante Svensson (seconda gobba, tau2 = 8): anch'esso AR(1), quindi l'errore
# di Nelson-Siegel cambia nel tempo e non si cancella togliendo una media
FATTORE_SVENSSON = {"media": 2.0, "phi": 0.99, "sigma": 0.10, "tau2": 8.0}


def genera_beta(giorni: int, rng) -> np.ndarray:
    """Tre fattori AR(1) indipendenti, partenza dalla media."""
    media, phi, sigma = (np.array(FATTORI[k]) for k in ("media", "phi", "sigma"))
    b = np.empty((giorni, 3))
    b[0] = media
    for t in range(1, giorni):
        b[t] = media + phi * (b[t - 1] - media) + sigma * rng.standard_normal(3)
    return b


def genera_ou(giorni: int, colonne: int, std_bp: float, emivita: float, rng) -> np.ndarray:
    """Mispricing OU in punti base, stazionario con la deviazione standard indicata."""
    if std_bp == 0:
        return np.zeros((giorni, colonne))
    phi = 0.5 ** (1 / emivita)
    m = np.empty((giorni, colonne))
    m[0] = std_bp * rng.standard_normal(colonne)
    inn = std_bp * np.sqrt(1 - phi**2)
    for t in range(1, giorni):
        m[t] = phi * m[t - 1] + inn * rng.standard_normal(colonne)
    return m


def genera_panel(giorni: int, std_bp: float, emivita: float, rumore_bp: float, rng, vero: str = "ns"):
    """Restituisce (rendimenti osservati [giorni x scadenze] in %, mispricing in bp, rumore di misura in bp, beta veri).

    vero = "ns": la curva vera e' Nelson-Siegel (il modello e' corretto);
    vero = "svensson": ha una seconda gobba (quarto fattore AR(1), tau2 = 8) che Nelson-Siegel non sa descrivere.
    """
    beta = genera_beta(giorni, rng)
    x = cv.carichi(SCADENZE, TAU_VERO)
    y = beta @ x.T
    if vero == "svensson":
        f = FATTORE_SVENSSON
        b3 = np.empty(giorni)
        b3[0] = f["media"]
        for t in range(1, giorni):
            b3[t] = f["media"] + f["phi"] * (b3[t - 1] - f["media"]) + f["sigma"] * rng.standard_normal()
        y = y + np.outer(b3, cv.carichi(SCADENZE, TAU_VERO, f["tau2"])[:, 3])
    elif vero != "ns":
        raise ValueError("vero deve essere 'ns' o 'svensson'")
    mis = genera_ou(giorni, len(SCADENZE), std_bp, emivita, rng)
    rumore = rumore_bp * rng.standard_normal(y.shape)
    return y + (mis + rumore) / 100, mis, rumore, beta


def guadagni(residui_bp: np.ndarray, residui_eseguibili_bp: np.ndarray, finestra: int = 60,
             soglia: float = 1.0, costo_bp: float = 0.0) -> np.ndarray:
    """Guadagno giornaliero medio (bp) sulle scadenze della strategia sul residuo.

    Segnale: z = (residuo - media mobile) / dev. std mobile sui `finestra` giorni PRECEDENTI
    (solo passato). Posizione al giorno t: +1 se z > soglia (residuo alto = titolo a
    buon mercato: si compra), -1 se z < -soglia, 0 altrimenti. Guadagno al giorno t+1 =
    posizione * (r_t - r_{t+1}) con r = residuo eseguibile (senza errore di misura); costo =
    costo_bp per ogni variazione di una unita' di posizione.
    """
    pos = strategia.posizioni(residui_bp, finestra, soglia)
    e = np.asarray(residui_eseguibili_bp, dtype=float)
    lordo = pos[:-1] * (e[:-1] - e[1:])
    cambi = np.abs(np.diff(pos, axis=0, prepend=0.0))[:-1]
    return (lordo - costo_bp * cambi).mean(axis=1)


def simula(giorni, std_bp, emivita, rumore_bp, seed, vero="ns", costo_bp=0.0, soglia=1.0, tau1=None, eseguibile=True):
    """Una simulazione completa: panel, fit giornaliero, strategia. Restituisce un dizionario.

    eseguibile=False calcola i guadagni sui residui OSSERVATI, rumore di misura compreso: e' quello che si fa
    sui dati reali, dove il rumore dei punti del Tesoro non si puo' separare. Serve da controllo nullo."""
    rng = np.random.default_rng(seed)
    y, mis, rumore, _ = genera_panel(giorni, std_bp, emivita, rumore_bp, rng, vero)
    beta, tau, res = cv.residui_panel(y, SCADENZE, tau1=tau1)
    res_bp = res * 100
    g = guadagni(res_bp, res_bp - rumore if eseguibile else res_bp, soglia=soglia, costo_bp=costo_bp)
    media, t = st.newey_west(g, ritardi=10)
    return {"media_bp": media, "t": t, "std_residuo_bp": float(res_bp.std()),
            "corr_residuo_mis": float(np.corrcoef(res_bp.ravel(), mis.ravel())[0, 1]) if std_bp > 0 else np.nan}


def potere(giorni, std_bp, emivita, rumore_bp, simulazioni, costo_bp=0.0, vero="ns", seme=0, soglia_t=2.0, tau1=None):
    """Frazione di simulazioni in cui il guadagno medio e' positivo con t di Newey-West > soglia_t,
    e guadagno medio sulle simulazioni. Con std_bp = 0 (nessun mispricing) e' la frequenza dei falsi positivi."""
    esiti = [simula(giorni, std_bp, emivita, rumore_bp, seme + i, vero, costo_bp, tau1=tau1) for i in range(simulazioni)]
    t = np.array([e["t"] for e in esiti])
    m = np.array([e["media_bp"] for e in esiti])
    return float(np.mean((t > soglia_t) & (m > 0))), float(m.mean())


def recupero(giorni, rumore_bp, seed, tau1=None):
    """Errore di stima senza mispricing: dev. std (rmse) dei beta, di tau e della curva (bp).

    Mostra che i parametri si stimano male (e' il caso dei beta della Fed) mentre la curva si stima bene."""
    rng = np.random.default_rng(seed)
    y, _, _, beta_vero = genera_panel(giorni, 0.0, 5.0, rumore_bp, rng)
    beta, tau, res = cv.residui_panel(y, SCADENZE, tau1=tau1)
    curva_vera = beta_vero @ cv.carichi(SCADENZE, TAU_VERO)[:, :3].T
    curva_stimata = y - res
    return {
        "rmse_beta0": float(np.sqrt(np.mean((beta[:, 0] - beta_vero[:, 0]) ** 2))),
        "rmse_beta1": float(np.sqrt(np.mean((beta[:, 1] - beta_vero[:, 1]) ** 2))),
        "rmse_beta2": float(np.sqrt(np.mean((beta[:, 2] - beta_vero[:, 2]) ** 2))),
        "rmse_tau": float(np.sqrt(np.mean((tau - TAU_VERO) ** 2))),
        "rmse_curva_bp": float(np.sqrt(np.mean((curva_stimata - curva_vera) ** 2)) * 100),
    }

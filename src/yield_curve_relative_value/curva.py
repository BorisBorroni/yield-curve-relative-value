"""Curva dei rendimenti di Nelson-Siegel e Svensson.

Unita': scadenze in anni, tassi in percentuale. I tassi zero sono a capitalizzazione
continua (come SVENY della Fed); il par yield e' invece il rendimento di una cedola
semestrale che fa valere il titolo 100 (come SVENPY): sono due cose diverse.

Formula (Svensson; Nelson-Siegel e' il caso beta3 = 0):
    y(n) = b0 + b1 * A(n/t1) + b2 * (A(n/t1) - exp(-n/t1)) + b3 * (A(n/t2) - exp(-n/t2))
    A(x) = (1 - exp(-x)) / x
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import least_squares

from . import obbligazioni as ob

GIORNI_ANNO = 365.25


@dataclass(frozen=True)
class Parametri:
    """Parametri della curva. Con beta3 = 0 la curva e' Nelson-Siegel (tau2 non conta)."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    tau1: float = 1.0
    tau2: float = 5.0

    def __post_init__(self):
        if not (self.tau1 > 0 and self.tau2 > 0):
            raise ValueError("tau1 e tau2 devono essere positivi")

    @property
    def ns(self) -> bool:
        return self.beta3 == 0.0


def _a(x):
    """A(x) = (1 - exp(-x)) / x, con A(0) = 1."""
    x = np.asarray(x, dtype=float)
    piccolo = np.abs(x) < 1e-8
    sicuro = np.where(piccolo, 1.0, x)
    return np.where(piccolo, 1.0 - x / 2, (1 - np.exp(-sicuro)) / sicuro)


def carichi(n, tau1: float, tau2: float | None = None) -> np.ndarray:
    """Matrice dei carichi (una riga per scadenza): [1, A1, A1 - e1] e, se c'e' tau2, [.., A2 - e2]."""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    x1 = n / tau1
    colonne = [np.ones_like(n), _a(x1), _a(x1) - np.exp(-x1)]
    if tau2 is not None:
        x2 = n / tau2
        colonne.append(_a(x2) - np.exp(-x2))
    return np.column_stack(colonne)


def rendimento_zero(n, p: Parametri) -> np.ndarray:
    """Tasso zero continuo (%) alla scadenza n."""
    tau2 = None if p.ns else p.tau2
    beta = [p.beta0, p.beta1, p.beta2] + ([] if p.ns else [p.beta3])
    return carichi(n, p.tau1, tau2) @ np.array(beta)


def fattore_sconto(n, p: Parametri) -> np.ndarray:
    """Fattore di sconto P(n) = exp(-y(n) n / 100)."""
    n = np.asarray(n, dtype=float)
    return np.exp(-rendimento_zero(n, p) * n / 100)


def forward(n, p: Parametri) -> np.ndarray:
    """Tasso forward istantaneo (%) a n anni."""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    x1 = n / p.tau1
    f = p.beta0 + p.beta1 * np.exp(-x1) + p.beta2 * x1 * np.exp(-x1)
    if not p.ns:
        x2 = n / p.tau2
        f = f + p.beta3 * x2 * np.exp(-x2)
    return f


def par_yield(scadenza: float, p: Parametri) -> float:
    """Par yield (%) a cedola semestrale per un titolo che scade fra `scadenza` anni.

    c = 200 (1 - P(T)) / somma P(t_i), t_i = 0.5, 1, ..., T. La scadenza deve essere
    multipla di 0.5.
    """
    k = int(round(scadenza * 2))
    if k < 1 or abs(k / 2 - scadenza) > 1e-9:
        raise ValueError("la scadenza deve essere un multiplo di 0.5 anni")
    t = np.arange(1, k + 1) / 2
    d = fattore_sconto(t, p)
    return float(200 * (1 - d[-1]) / d.sum())


def prezzo_sporco_modello(cedola: float, scadenza: date, regolamento: date, p: Parametri) -> float:
    """Prezzo sporco del titolo scontando i suoi flussi con la curva (tempi ACT/365.25)."""
    f = ob.flussi(cedola, scadenza, regolamento)
    t = np.array([(d - regolamento).days / GIORNI_ANNO for d, _ in f])
    importi = np.array([x for _, x in f])
    return float(importi @ fattore_sconto(t, p))


@dataclass(frozen=True)
class Titolo:
    """Titolo osservato: cedola (%), scadenza, prezzo sporco di mercato."""

    cedola: float
    scadenza: date
    prezzo_sporco: float


def _pesi(titoli: list[Titolo], regolamento: date) -> np.ndarray:
    """Pesi 1/duration modificata (come Gurkaynak-Sack-Wright): un errore di prezzo
    su un titolo lungo vale meno di uno su un titolo corto, a parita' di errore di rendimento."""
    pesi = []
    for t in titoli:
        y = ob.rendimento(t.prezzo_sporco, t.cedola, t.scadenza, regolamento, pulito=False)
        pesi.append(1 / ob.duration_modificata(y, t.cedola, t.scadenza, regolamento))
    return np.array(pesi)


def calibra_prezzi(
    titoli: list[Titolo],
    regolamento: date,
    tau1: float | None = None,
    tau2: float | None = None,
    beta_iniziali: tuple[float, float, float] = (4.0, -1.0, -1.0),
) -> Parametri:
    """Calibra la curva sui prezzi sporchi minimizzando gli errori di prezzo pesati 1/duration.

    - tau1 fisso (Diebold-Li): si stimano solo i beta; tau1 libero: si stima anche tau1.
    - tau2 assente: Nelson-Siegel; tau2 fisso: Svensson con beta3 libero.
    Il caso tau1 libero e tau2 assente e' il Nelson-Siegel completo.
    """
    pesi = _pesi(titoli, regolamento)
    usa_svensson = tau2 is not None
    tau1_libero = tau1 is None

    def costruisci(x):
        beta = x[:4] if usa_svensson else x[:3]
        t1 = x[-1] if tau1_libero else tau1
        return Parametri(beta[0], beta[1], beta[2], beta[3] if usa_svensson else 0.0,
                         t1, tau2 if usa_svensson else 5.0)

    def residui(x):
        p = costruisci(x)
        mod = np.array([prezzo_sporco_modello(t.cedola, t.scadenza, regolamento, p) for t in titoli])
        mkt = np.array([t.prezzo_sporco for t in titoli])
        return (mod - mkt) * pesi

    x0 = list(beta_iniziali) + ([0.0] if usa_svensson else []) + ([2.0] if tau1_libero else [])
    inf = [-np.inf] * (4 if usa_svensson else 3) + ([0.1] if tau1_libero else [])
    sup = [np.inf] * (4 if usa_svensson else 3) + ([20.0] if tau1_libero else [])
    sol = least_squares(residui, x0, bounds=(inf, sup), x_scale=1.0)
    return costruisci(sol.x)


def adatta_rendimenti(scadenze, rendimenti, tau1: float | None = None, griglia=None) -> Parametri:
    """Nelson-Siegel su tassi zero gia' noti (non su prezzi).

    Con tau1 fisso e' una regressione lineare (Diebold-Li). Con tau1 libero si prova una
    griglia di valori e per ciascuno i beta si ottengono per minimi quadrati: si tiene il
    tau con errore minimo (si evitano i minimi locali della stima non lineare).
    """
    n = np.asarray(scadenze, dtype=float)
    y = np.asarray(rendimenti, dtype=float)
    candidati = [tau1] if tau1 is not None else (griglia if griglia is not None else np.linspace(0.2, 10, 99))
    migliore = None
    for t in candidati:
        x = carichi(n, t)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        errore = float(np.sum((x @ beta - y) ** 2))
        if migliore is None or errore < migliore[0]:
            migliore = (errore, t, beta)
    _, t, beta = migliore
    return Parametri(float(beta[0]), float(beta[1]), float(beta[2]), 0.0, float(t))


def condizionamento(scadenze, tau1: float, tau2: float | None = None) -> float:
    """Indice di condizionamento della matrice dei carichi: se e' molto alto (oltre circa 1e3)
    i beta non sono identificabili con quelle scadenze (colonne quasi collineari)."""
    return float(np.linalg.cond(carichi(scadenze, tau1, tau2)))


def residui_panel(rendimenti, scadenze, tau1: float | None = None, griglia=None):
    """Nelson-Siegel giorno per giorno su una tabella di rendimenti (righe = giorni, colonne = scadenze).

    Per ogni giorno si sceglie il tau (sulla griglia) che minimizza la somma dei quadrati e i beta
    sono quelli dei minimi quadrati con quel tau. Restituisce (beta [giorni x 3], tau [giorni],
    residui [giorni x scadenze]) con residuo = osservato - curva. Le righe con valori mancanti
    non sono ammesse: chi chiama seleziona le colonne disponibili.
    """
    y = np.asarray(rendimenti, dtype=float)
    n = np.asarray(scadenze, dtype=float)
    if np.isnan(y).any():
        raise ValueError("residui_panel non accetta valori mancanti: seleziona prima le colonne disponibili")
    candidati = [tau1] if tau1 is not None else (griglia if griglia is not None else np.geomspace(0.3, 10, 40))
    migliore_ssr = np.full(len(y), np.inf)
    beta = np.zeros((len(y), 3))
    tau = np.zeros(len(y))
    residui = np.zeros_like(y)
    for t in candidati:
        x = carichi(n, t)
        pinv = np.linalg.pinv(x)  # 3 x scadenze
        b = y @ pinv.T
        r = y - b @ x.T
        ssr = np.sum(r**2, axis=1)
        meglio = ssr < migliore_ssr
        migliore_ssr[meglio] = ssr[meglio]
        beta[meglio] = b[meglio]
        tau[meglio] = t
        residui[meglio] = r[meglio]
    return beta, tau, residui


def par_yield_vettore(scadenze, p: Parametri) -> np.ndarray:
    """Par yield (%) per piu' scadenze, tutte multiple di 0.5 anni (stessa formula di par_yield)."""
    k = np.rint(np.asarray(scadenze, dtype=float) * 2).astype(int)
    if np.any(k < 1) or np.any(np.abs(k / 2 - np.asarray(scadenze)) > 1e-9):
        raise ValueError("le scadenze devono essere multipli di 0.5 anni")
    d = fattore_sconto(np.arange(1, k.max() + 1) / 2, p)
    somma = np.cumsum(d)
    return 200 * (1 - d[k - 1]) / somma[k - 1]


def adatta_par_yield(scadenze, osservati, svensson: bool = False) -> Parametri:
    """Adatta la curva (zero di Nelson-Siegel o Svensson) in modo che i suoi par yield
    riproducano i par yield osservati (minimi quadrati sugli errori in punti percentuali).

    I punti del Tesoro sono par yield, non tassi zero: adattare la formula degli zero direttamente
    ai par yield darebbe un errore sistematico che cresce con la scadenza. I valori mancanti (NaN)
    sono ignorati; servono almeno tanti punti quanti parametri.

    La stima e' non lineare e ha minimi locali: si parte da piu' punti iniziali (tau diversi) e si
    tiene la soluzione migliore. Ripartire ogni giorno dalla soluzione del giorno prima peggiorava
    il risultato, perche' una soluzione degenere (tau al limite) si trascinava nei giorni dopo.
    Limiti: tau1 in [0.3, 10] anni.
    """
    n = np.asarray(scadenze, dtype=float)
    y = np.asarray(osservati, dtype=float)
    ok = ~np.isnan(y)
    n, y = n[ok], y[ok]
    npar = 6 if svensson else 4
    if len(y) < npar:
        raise ValueError("pochi punti per i parametri da stimare")

    def costruisci(x):
        if svensson:
            return Parametri(x[0], x[1], x[2], x[3], x[4], x[5])
        return Parametri(x[0], x[1], x[2], 0.0, x[3], 5.0)

    def residui(x):
        return par_yield_vettore(n, costruisci(x)) - y

    inf = [-50, -50, -50] + ([-50, 0.3, 0.3] if svensson else [0.3])
    sup = [50, 50, 50] + ([50, 10, 30] if svensson else [10])
    migliore = None
    for tau in (1.0, 3.0, 8.0):
        coda = [0.0, tau, 2 * tau + 2] if svensson else [tau]
        x0 = np.clip([y[-1], y[0] - y[-1], 0.0, *coda], np.array(inf) + 1e-6, np.array(sup) - 1e-6)
        sol = least_squares(residui, x0, bounds=(inf, sup))
        if migliore is None or sol.cost < migliore.cost:
            migliore = sol
    return costruisci(migliore.x)

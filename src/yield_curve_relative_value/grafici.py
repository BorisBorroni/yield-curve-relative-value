"""Grafici del progetto (matplotlib). Colori e stile: un solo asse, linee, etichette dirette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analisi as an  # noqa: E402
from . import strategia as sg  # noqa: E402

SFONDO, INCHIOSTRO, SECONDARIO, GRIGLIA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORI = ["#2a78d6", "#eb6834", "#4a3aa7"]  # blu, arancione, viola 


def _stile(ax, titolo, ylabel):
    ax.set_facecolor(SFONDO)
    ax.set_title(titolo, loc="left", fontsize=11, color=INCHIOSTRO, pad=10)
    ax.set_ylabel(ylabel, color=SECONDARIO, fontsize=9)
    ax.grid(axis="y", color=GRIGLIA, linewidth=0.8)
    ax.tick_params(colors=SECONDARIO, labelsize=9)
    for lato in ("top", "right", "left"):
        ax.spines[lato].set_visible(False)
    ax.spines["bottom"].set_color(GRIGLIA)
    ax.axvline(an.CAMBIO_METODO, color=SECONDARIO, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(an.CAMBIO_METODO, ax.get_ylim()[1], " 6 dic 2021: nuovo metodo del Tesoro",
            color=SECONDARIO, fontsize=8, va="top", ha="left")


def _etichette(ax, serie):
    """Etichetta diretta all'estremo destro di ogni linea e legenda in basso."""
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=SECONDARIO)
    for nome, s in serie:
        s = s.dropna()
        ax.annotate(nome, (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INCHIOSTRO, fontsize=9, va="center")


def grafico_premio(premio_bp: pd.DataFrame, scadenze=(10, 20, 30), finestra: int = 250):
    """Premio del titolo nuovo: Tesoro meno Fed (bp), media mobile di un anno di borsa, per alcune scadenze."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SFONDO)
    serie = []
    for c, col in zip(scadenze, COLORI, strict=False):
        s = premio_bp[c].rolling(finestra, min_periods=finestra // 2).mean()
        ax.plot(s.index, s, color=col, linewidth=1.8, label=f"{c} anni")
        serie.append((f"{c} anni", s))
    ax.axhline(0, color=SECONDARIO, linewidth=0.8)
    ax.set_ylim(min(-25, np.nanmin([s.min() for _, s in serie]) - 2), max(10, np.nanmax([s.max() for _, s in serie]) + 2))
    _stile(ax, "Tesoro meno curva Fed (media mobile di un anno)", "punti base")
    _etichette(ax, serie)
    ax.set_xlim(premio_bp.index[0], premio_bp.index[-1] + pd.Timedelta(days=500))
    fig.tight_layout()
    return fig


def guadagni_cumulati(x: pd.Series, somma_pesi: float, casi=((0.0, 0), (0.0, 1), (0.25, 1))) -> pd.DataFrame:
    """Guadagno cumulato (bp) per ogni caso (costo in bp, ritardo di esecuzione in giorni)."""
    out = {}
    for costo, ritardo in casi:
        g, _ = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), somma_pesi, 60, 1.0, costo, ritardo=ritardo)
        nome = ("lordo" if costo == 0 else f"costo {costo:g} bp") + (", giorno dopo" if ritardo else ", stesso giorno")
        out[nome] = pd.Series(g, index=x.index[1 + ritardo :]).cumsum()
    return pd.DataFrame(out)


def grafico_guadagni(cum: pd.DataFrame, titolo: str):
    """Guadagno cumulato: lordo allo stesso prezzo del segnale, lordo il giorno dopo, e con un costo di 0.25 bp."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SFONDO)
    serie = []
    for c, col in zip(cum.columns, COLORI, strict=False):
        ax.plot(cum.index, cum[c], color=col, linewidth=1.8, label=c)
        serie.append((c, cum[c]))
    ax.axhline(0, color=SECONDARIO, linewidth=0.8)
    _stile(ax, titolo, "punti base cumulati")
    _etichette(ax, serie)
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=700))
    fig.tight_layout()
    return fig

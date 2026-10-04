"""Grafici del progetto (matplotlib). Colori e stile: un solo asse, linee sottili, etichette dirette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analisi as an  # noqa: E402
from . import strategia as sg  # noqa: E402

SFONDO, INCHIOSTRO, SECONDARIO, GRIGLIA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORI = ["#2a78d6", "#eb6834", "#4a3aa7"]  # blu, arancione, viola (primi colori della palette validata)


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


def _etichette(ax, serie, colori):
    """Etichetta diretta all'estremo destro di ogni linea e legenda in basso."""
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=SECONDARIO)
    for (nome, s), c in zip(serie, colori, strict=True):
        s = s.dropna()
        ax.annotate(nome, (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INCHIOSTRO, fontsize=9, va="center")
        ax.plot([], [], color=c)


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
    _etichette(ax, serie, COLORI)
    ax.set_xlim(premio_bp.index[0], premio_bp.index[-1] + pd.Timedelta(days=500))
    fig.tight_layout()
    return fig


def guadagni_cumulati(x: pd.Series, somma_pesi: float, costi=(0.0, 0.25, 0.5)) -> pd.DataFrame:
    """Guadagno cumulato (bp) della regola z-score su una serie, per diversi costi."""
    out = {}
    for c in costi:
        g, _ = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), somma_pesi, 60, 1.0, c)
        out[c] = pd.Series(g, index=x.index[1:]).cumsum()
    return pd.DataFrame(out)


def grafico_guadagni(cum: pd.DataFrame, titolo: str):
    """Guadagno cumulato per costo: senza costi la regola sembra funzionare, con costi realistici no."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SFONDO)
    serie = []
    for c, col in zip(cum.columns, COLORI, strict=False):
        nome = f"costo {c:g} bp"
        ax.plot(cum.index, cum[c], color=col, linewidth=1.8, label=nome)
        serie.append((nome, cum[c]))
    ax.axhline(0, color=SECONDARIO, linewidth=0.8)
    _stile(ax, titolo, "punti base cumulati")
    _etichette(ax, serie, COLORI)
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=700))
    fig.tight_layout()
    return fig

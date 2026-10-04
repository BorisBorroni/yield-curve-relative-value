"""Livello 5: valore relativo sulla curva. Ipotesi e parametri dichiarati PRIMA di guardare i risultati.

Ipotesi (al massimo quattro; i p-value non sono corretti per test multipli):
  H1 Errore di modello. I residui di Nelson-Siegel sui punti del Tesoro contengono una forma sistematica che
     Svensson cattura: il rapporto fra lo scarto di Svensson e quello di Nelson-Siegel e' piu' basso di quello
     che darebbe il solo rumore (nel mondo sintetico con curva Nelson-Siegel vera il rapporto e' circa
     radice(2/4) = 0.71, perche' si passa da 4 a 6 parametri su 8 punti).
  H2 Segnale sui residui. Il z-score dei residui di Nelson-Siegel (finestra 60 giorni, soglia 1) predice il
     rientro: il guadagno medio per giorno, al netto di un costo di 0.25 bp, e' positivo con t di Newey-West > 2.
  H3 Farfalle. Le farfalle 2-5-10 e 5-10-30, con la stessa regola, danno un guadagno netto (costo 0.25 bp)
     positivo con t > 2, e la versione neutrale ai fattori (livello e pendenza) fa meglio di quella con ali 50/50.
  H4 Stabilita'. Il segno del guadagno netto e' lo stesso prima e dopo il 6 dicembre 2021.

Parametri fissati prima: finestra 60 giorni, soglia 1, costi 0 / 0.25 / 0.5 bp, tau di Diebold-Li 1.37 anni,
Newey-West con 10 ritardi. Niente di questo e' stato scelto guardando i dati: i parametri non sono stimati
sui dati, quindi ogni giorno e' gia' fuori campione. L'analisi di sensibilita' mostra cosa cambia variando
finestra e soglia, non serve a sceglierli.

ATTENZIONE: i punti del Tesoro sono letture interpolate e non prezzi eseguibili. Il loro rumore di misura
rientra da solo e gonfia i guadagni di una strategia di ritorno alla media. I test sono quindi spostati a
favore di un guadagno positivo: un risultato negativo e' solido, uno positivo non basterebbe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import analisi as an  # noqa: E402
from yield_curve_relative_value import curva as cv  # noqa: E402
from yield_curve_relative_value import dati  # noqa: E402
from yield_curve_relative_value import mondo_sintetico as ms  # noqa: E402
from yield_curve_relative_value import statistica as stat  # noqa: E402
from yield_curve_relative_value import strategia as sg  # noqa: E402

OUT = ROOT / "output"
FINESTRA, SOGLIA, COSTI, TAU, RITARDI = 60, 1.0, (0.0, 0.25, 0.5), an.TAU_DL, 10
FARFALLE = {"2-5-10": (2, 5, 10), "5-10-30": (5, 10, 30)}
pd.set_option("display.width", 170)


def da_cache(nome, calcolo):
    f = OUT / nome
    if f.exists():
        risultato = pd.read_csv(f, index_col=0, parse_dates=True)
        risultato.columns = [int(c) for c in risultato.columns]
        return risultato
    risultato = calcolo()
    risultato.to_csv(f)
    return risultato


def tratto_continuo(df: pd.DataFrame) -> pd.DataFrame:
    """Dal giorno dopo l'ultimo dato mancante di una colonna (i 30 anni mancano dal 2002 al 2006)."""
    mancanti = df.index[df.isna().any(axis=1)]
    return df if len(mancanti) == 0 else df[df.index > mancanti[-1]]


def per_periodo(serie: pd.Series) -> dict[str, pd.Series]:
    return {"intero": serie, "fino a dic 2021": serie[serie.index < an.CAMBIO_METODO],
            "da dic 2021": serie[serie.index >= an.CAMBIO_METODO]}


def riga(g: pd.Series) -> dict:
    m, t = stat.newey_west(g.to_numpy(), RITARDI)
    return {"bp/giorno": round(m, 4), "bp/anno": round(m * 252, 1), "t_NW": round(t, 2),
            "Sharpe": round(m / g.std() * np.sqrt(252), 2) if g.std() > 0 else np.nan, "giorni": len(g)}


def rapporto(ns, sv, maschera):
    a, b = ns.loc[maschera], sv.loc[maschera]
    return np.sqrt((b**2).mean() / (a**2).mean()), float(np.sqrt(np.nanmean(b.to_numpy() ** 2) / np.nanmean(a.to_numpy() ** 2)))


def h1(par_t):
    ns = da_cache("ns_residui_bp.csv", lambda: an.adatta_serie(par_t)[1])
    sv = da_cache("svensson_residui_bp.csv", lambda: an.adatta_serie(par_t, svensson=True)[1])
    prima, dopo = ns.index < an.CAMBIO_METODO, ns.index >= an.CAMBIO_METODO
    periodi = {"intero": np.ones(len(ns), bool), "fino a dic 2021": prima, "da dic 2021": dopo}
    tab = pd.DataFrame({k: rapporto(ns, sv, m)[0] for k, m in periodi.items()}).T.round(2)
    tab["tutte"] = [round(rapporto(ns, sv, m)[1], 3) for m in periodi.values()]
    print("\n== H1: scarto di Svensson / scarto di Nelson-Siegel (dati reali) ==")
    print(tab.to_string())
    # riferimento: curva Nelson-Siegel vera con solo rumore di 3 bp, 120 giorni
    rng = np.random.default_rng(11)
    n = np.array(dati.SCADENZE, float)
    vero = cv.par_yield_vettore(n, cv.Parametri(4.5, -2.0, -1.0, 0.0, 2.5))
    sim = pd.DataFrame(vero + rng.normal(0, 0.03, (120, len(n))), columns=n)
    r_ns = an.adatta_serie(sim)[1].to_numpy()
    r_sv = an.adatta_serie(sim, svensson=True)[1].to_numpy()
    rif = float(np.sqrt((r_sv**2).mean() / (r_ns**2).mean()))
    print(f"Riferimento sintetico (curva NS vera, solo rumore di 3 bp): {rif:.3f}")
    tab.to_csv(OUT / "h1_rapporto_svensson.csv")
    return rif


def h2(residui):
    righe = []
    r = residui.to_numpy()
    for costo in COSTI:
        cols = [sg.guadagni_serie(r[:, j], r[:, j], 1.0, FINESTRA, SOGLIA, costo)[0] for j in range(r.shape[1])]
        g = pd.Series(np.nanmean(np.column_stack([np.nan_to_num(c) for c in cols]), axis=1), index=residui.index[1:])
        for p, s in per_periodo(g).items():
            righe.append({"costo (bp)": costo, "periodo": p, **riga(s)})
    return pd.DataFrame(righe).set_index(["costo (bp)", "periodo"])


def strategie_farfalle(par_t):
    out = {}
    for nome, terna in FARFALLE.items():
        base = tratto_continuo(par_t[list(terna)])
        for neutralita in ("dv01", "fattori"):
            w = sg.pesi_farfalla(terna, neutralita, TAU)
            out[(nome, neutralita)] = (sg.serie_farfalla(base, terna, w), w)
    return out


def h3(farf):
    righe = []
    for (nome, neutr), (x, w) in farf.items():
        for costo in COSTI:
            g, pos = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), np.abs(w).sum(), FINESTRA, SOGLIA, costo)
            gs = pd.Series(g, index=x.index[1:])
            for p, s in per_periodo(gs).items():
                if len(s) > 100:
                    righe.append({"farfalla": nome, "neutralita": neutr, "pesi": np.round(w, 2).tolist(),
                                  "costo (bp)": costo, "periodo": p, **riga(s)})
    return pd.DataFrame(righe).set_index(["farfalla", "neutralita", "costo (bp)", "periodo"])


def sensibilita(farf):
    righe = []
    for (nome, neutr), (x, w) in farf.items():
        for fin in (20, 60, 120):
            for sog in (0.5, 1.0, 1.5, 2.0):
                g, _ = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), np.abs(w).sum(), fin, sog, 0.25)
                righe.append({"farfalla": f"{nome} {neutr}", "finestra": fin, "soglia": sog, "bp/giorno netti": round(g.mean(), 4)})
        g, _ = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), np.abs(w).sum(), FINESTRA, SOGLIA, 0.25, emivita_max=60)
        righe.append({"farfalla": f"{nome} {neutr}", "finestra": 60, "soglia": 1.0, "bp/giorno netti": round(g.mean(), 4), "filtro": "emivita<=60"})
    t = pd.DataFrame(righe)
    return t[t["filtro"].isna()].pivot_table(index=["farfalla", "finestra"], columns="soglia", values="bp/giorno netti") if "filtro" in t else t, t


def rotazione(farf):
    righe = []
    for (nome, neutr), (x, w) in farf.items():
        g, pos = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), np.abs(w).sum(), FINESTRA, SOGLIA, 0.0)
        scambiato = np.abs(w).sum() * np.abs(np.diff(pos, prepend=0.0)).mean()  # DV01 scambiato per giorno
        righe.append({"farfalla": f"{nome} {neutr}", "giorni in posizione (%)": round(100 * (pos != 0).mean(), 1),
                      "cambi di posizione per giorno": round(np.abs(np.diff(pos)).mean(), 3),
                      "costo di pareggio (bp)": round(g.mean() / scambiato, 3),
                      "dev. std serie (bp)": round(x.std(), 1)})
    return pd.DataFrame(righe).set_index("farfalla")


def controllo_nullo():
    """Mondo sintetico SENZA mispricing, con guadagni calcolati sui residui osservati come sui dati reali.
    Mostra quanto guadagno 'fasullo' producono da soli il rumore di misura e l'errore di modello."""
    righe = []
    for nome, vero, rumore in (("curva NS vera, rumore 4 bp", "ns", 4.0), ("curva Svensson, rumore 1 bp", "svensson", 1.0)):
        for costo in COSTI:
            es = [ms.simula(1500, 0.0, 5.0, rumore, 700 + i, vero, costo, eseguibile=False) for i in range(20)]
            m = np.array([e["media_bp"] for e in es])
            righe.append({"mondo senza mispricing": nome, "costo (bp)": costo, "bp/giorno": round(m.mean(), 4),
                          "residuo (bp)": round(np.mean([e["std_residuo_bp"] for e in es]), 2),
                          "simulazioni con t>2": f"{int(np.sum([e['t'] > 2 for e in es]))}/20"})
    return pd.DataFrame(righe).set_index(["mondo senza mispricing", "costo (bp)"])


def main():
    OUT.mkdir(exist_ok=True)
    try:
        cmt, _ = dati.carica_storico()
    except FileNotFoundError as e:
        sys.exit(str(e))
    par_t = dati.par_cmt(cmt).dropna(how="all")
    rif = h1(par_t)
    residui = da_cache("ns_residui_bp.csv", lambda: an.adatta_serie(par_t)[1])

    t2 = h2(residui)
    print("\n== H2: strategia sui residui di Nelson-Siegel (media delle 8 scadenze, costo su una sola gamba) ==")
    print(t2.to_string())
    t2.to_csv(OUT / "h2_residui.csv")

    nullo = controllo_nullo()
    print("\n== Controllo nullo: guadagni sui residui osservati in un mondo senza alcun mispricing ==")
    print(nullo.to_string())
    nullo.to_csv(OUT / "controllo_nullo.csv")

    farf = strategie_farfalle(par_t)
    t3 = h3(farf)
    print("\n== H3 e H4: farfalle ==")
    print(t3.to_string())
    t3.to_csv(OUT / "h3_farfalle.csv")
    print("\n== Rotazione (costo 0) ==")
    print(rotazione(farf).to_string())
    piv, tutto = sensibilita(farf)
    print("\n== Sensibilita' (bp/giorno netti, costo 0.25) ==")
    print(piv.to_string())
    print("\nFiltro emivita <= 60 giorni:")
    print(tutto[tutto["filtro"].notna()][["farfalla", "bp/giorno netti"]].to_string(index=False))
    tutto.to_csv(OUT / "sensibilita_farfalle.csv")
    print(f"\n(Riferimento H1 sintetico: {rif:.3f})")


if __name__ == "__main__":
    main()

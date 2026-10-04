"""Valore relativo sulla curva (sezione 9 del README).

Ipotesi (al massimo quattro; i p-value non sono corretti per test multipli):
  H1 Errore di modello. I residui di Nelson-Siegel sui punti del Tesoro contengono una forma sistematica che
     Svensson cattura: il rapporto fra lo scarto di Svensson e quello di Nelson-Siegel e' piu' basso di quello
     che darebbe il solo rumore (in una simulazione con curva Nelson-Siegel vera e solo rumore di 3 bp il rapporto e' circa 0.75, e
     per scadenza e' riportato sotto; la stima teorica radice(2/4) = 0.71 e' solo indicativa).
  H2 Segnale sui residui. Il z-score dei residui di Nelson-Siegel (finestra 60 giorni, soglia 1) predice il
     rientro: il guadagno medio per giorno, al netto di un costo di 0.25 bp, e' positivo con t di Newey-West > 2.
  H3 Farfalle. Le farfalle 2-5-10 e 5-10-30, con la stessa regola, danno un guadagno netto (costo 0.25 bp)
     positivo con t > 2, e la versione neutrale ai fattori (livello e pendenza) fa meglio di quella con ali 50/50.
  H4 Stabilita'. Il segno del guadagno netto e' lo stesso prima e dopo il 6 dicembre 2021.

Parametri: finestra 60 giorni, soglia 1, costi 0 / 0.25 / 0.5 bp, tau di Diebold-Li 1.37 anni, Newey-West con
10 ritardi. Sono convenzionali e non sono stimati sui dati, ma non posso provare di averli fissati prima di
vedere i dati. Dopo una prima lettura dei risultati ho aggiunto tre controlli, senza cambiare i criteri
di H1-H4: l'esecuzione il giorno dopo il segnale, un controllo nullo calibrato sui residui reali, il t con
60 ritardi. La sensibilita' mostra cosa cambia variando finestra e soglia, non serve a sceglierle.

Attenzione: i punti del Tesoro sono letture interpolate e non prezzi eseguibili. Il loro rumore di misura
rientra da solo e gonfia i guadagni di una strategia di ritorno alla media, e con lo scambio al prezzo
stesso che genera il segnale il guadagno lordo e' sovrastimato (vedi il ritardo di un giorno). Un risultato
positivo va quindi guardato con piu' sospetto di uno negativo.
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
    g = g.dropna()
    m, t = stat.newey_west(g.to_numpy(), RITARDI)
    return {"bp/giorno": round(m, 4), "t_NW10": round(t, 2), "t_NW60": round(stat.newey_west(g.to_numpy(), 60)[1], 2),
            "giorni": len(g)}


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
    # SSR di Svensson contro Nelson-Siegel: non puo' essere peggiore, perche' Nelson-Siegel e' un caso particolare
    peggiori = int(((sv**2).sum(axis=1) > (ns**2).sum(axis=1) * (1 + 1e-9)).sum())
    print(f"Giorni in cui la somma dei quadrati di Svensson supera quella di Nelson-Siegel: {peggiori} su {len(ns)}")
    # riferimento: curva Nelson-Siegel vera e solo rumore di 3 bp, 120 giorni; aggregato con il seme 11, per
    # scadenza la media su 10 semi (11-20)
    n = np.array(dati.SCADENZE, float)
    vero = cv.par_yield_vettore(n, cv.Parametri(4.5, -2.0, -1.0, 0.0, 2.5))
    aggregati, per_scadenza = [], []
    for seme in range(11, 21):
        sim = pd.DataFrame(vero + np.random.default_rng(seme).normal(0, 0.03, (120, len(n))), columns=n)
        r_ns = an.adatta_serie(sim)[1].to_numpy()
        r_sv = an.adatta_serie(sim, svensson=True)[1].to_numpy()
        aggregati.append(float(np.sqrt((r_sv**2).mean() / (r_ns**2).mean())))
        per_scadenza.append(np.sqrt((r_sv**2).mean(axis=0) / (r_ns**2).mean(axis=0)))
    rif = aggregati[0]
    rif_scad = pd.Series(np.mean(per_scadenza, axis=0), index=[int(x) for x in n]).round(2)
    print(f"Riferimento sintetico (curva NS vera, solo rumore di 3 bp), seme 11: {rif:.3f}; "
          f"media su 10 semi: {np.mean(aggregati):.3f}")
    print("Riferimento per scadenza (media su 10 semi):", rif_scad.to_dict())
    rif_scad.to_frame("rapporto solo rumore").to_csv(OUT / "h1_riferimento_per_scadenza.csv")
    tab.to_csv(OUT / "h1_rapporto_svensson.csv")
    return rif


def h2(residui):
    """Media delle 8 scadenze del guadagno sul residuo (una gamba sola, costo su quella gamba). Le scadenze
    senza dato (30 anni, 2002-2006) escono dalla media del giorno. `ritardo` 1 = scambio il giorno dopo."""
    righe = []
    r = residui.to_numpy()
    for ritardo in (0, 1):
        for costo in COSTI:
            cols = [sg.guadagni_serie(r[:, j], r[:, j], 1.0, FINESTRA, SOGLIA, costo, ritardo=ritardo)[0]
                    for j in range(r.shape[1])]
            g = pd.Series(np.nanmean(np.column_stack(cols), axis=1), index=residui.index[1 + ritardo :])
            for p, s in per_periodo(g).items():
                righe.append({"esecuzione": "giorno dopo" if ritardo else "stesso giorno", "costo (bp)": costo,
                              "periodo": p, **riga(s)})
    return pd.DataFrame(righe).set_index(["esecuzione", "costo (bp)", "periodo"])


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
        for ritardo in (0, 1):
            for costo in COSTI:
                g, pos = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), np.abs(w).sum(), FINESTRA, SOGLIA, costo,
                                           ritardo=ritardo)
                gs = pd.Series(g, index=x.index[1 + ritardo :])
                for p, s in per_periodo(gs).items():
                    if len(s) > 100:
                        righe.append({"farfalla": nome, "neutralita": neutr, "pesi": np.round(w, 2).tolist(),
                                      "esecuzione": "giorno dopo" if ritardo else "stesso giorno",
                                      "costo (bp)": costo, "periodo": p, **riga(s)})
    return pd.DataFrame(righe).set_index(["farfalla", "neutralita", "esecuzione", "costo (bp)", "periodo"])


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


def controllo_nullo(residui, simulazioni=200, seme=2024):
    """Residui stazionari senza alcun mispricing, con la stessa persistenza e la stessa dev. std di quelli reali
    (AR(1) per scadenza, stimato sui dati), stessa lunghezza e stessa regola. Un residuo stazionario torna alla
    media per costruzione: quanto guadagna la regola in quel caso e' il termine di paragone del guadagno reale,
    non lo zero. Per ogni caso: media e 5-95 percentile dei guadagni simulati e quota di simulazioni
    che superano il valore reale."""
    r = residui.to_numpy()
    phi = np.array([np.corrcoef(c[~np.isnan(c)][:-1], c[~np.isnan(c)][1:])[0, 1] for c in r.T])
    sd = np.array([np.nanstd(c) for c in r.T])
    reale = {}
    for ritardo in (0, 1):
        for costo in (0.0, 0.25):
            cols = [sg.guadagni_serie(r[:, j], r[:, j], 1.0, FINESTRA, SOGLIA, costo, ritardo=ritardo)[0]
                    for j in range(r.shape[1])]
            reale[(ritardo, costo)] = float(np.nanmean(np.column_stack(cols)))
    rng = np.random.default_rng(seme)
    n = len(r)
    sim = {k: [] for k in reale}
    for _ in range(simulazioni):
        x = np.empty((n, len(phi)))
        x[0] = sd * rng.standard_normal(len(phi))
        inn = sd * np.sqrt(1 - phi**2)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + inn * rng.standard_normal(len(phi))
        for (ritardo, costo) in reale:
            g, _ = sg.guadagni_serie(x, x, 1.0, FINESTRA, SOGLIA, costo, ritardo=ritardo)
            sim[(ritardo, costo)].append(g.mean())
    righe = []
    for (ritardo, costo), v in sim.items():
        v = np.array(v)
        righe.append({"esecuzione": "giorno dopo" if ritardo else "stesso giorno", "costo (bp)": costo,
                      "reale (bp/giorno)": round(reale[(ritardo, costo)], 4),
                      "nullo, media": round(v.mean(), 4), "nullo, 5%": round(np.percentile(v, 5), 4),
                      "nullo, 95%": round(np.percentile(v, 95), 4),
                      "quota nulli >= reale": round(float(np.mean(v >= reale[(ritardo, costo)])), 3)})
    print("phi per scadenza:", np.round(phi, 3), " dev. std (bp):", np.round(sd, 2))
    return pd.DataFrame(righe).set_index(["esecuzione", "costo (bp)"])


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

    nullo = controllo_nullo(residui)
    print("\n== Controllo nullo: residui stazionari senza mispricing, stessa persistenza e dev. std dei reali ==")
    print(nullo.to_string())
    nullo.to_csv(OUT / "controllo_nullo.csv")

    farf = strategie_farfalle(par_t)
    t3 = h3(farf)
    print("\n== H3 e H4: farfalle ==")
    print(t3.to_string())
    t3.to_csv(OUT / "h3_farfalle.csv")
    print("\n== Rotazione (costo 0) ==")
    rot = rotazione(farf)
    print(rot.to_string())
    rot.to_csv(OUT / "rotazione_farfalle.csv")
    piv, tutto = sensibilita(farf)
    print("\n== Sensibilita' (bp/giorno netti, costo 0.25) ==")
    print(piv.to_string())
    print("\nFiltro emivita <= 60 giorni:")
    print(tutto[tutto["filtro"].notna()][["farfalla", "bp/giorno netti"]].to_string(index=False))
    tutto.to_csv(OUT / "sensibilita_farfalle.csv")
    print(f"\n(Riferimento H1 sintetico: {rif:.3f})")


if __name__ == "__main__":
    main()

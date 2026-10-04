"""Dati reali (sezione 8 del README): adatta la curva ogni giorno e produce le tabelle.

Legge lo storico (data/storico/), scrive in output/: parametri e residui del fit giornaliero e le
tabelle stampate. Il fit su circa 6500 giorni richiede 2-3 minuti.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import analisi as an  # noqa: E402
from yield_curve_relative_value import dati  # noqa: E402

OUT = ROOT / "output"
pd.set_option("display.width", 160)


def mostra(titolo, tabella, nome_file):
    print(f"\n== {titolo} ==")
    print(tabella.to_string())
    tabella.to_csv(OUT / f"{nome_file}.csv")


def main():
    OUT.mkdir(exist_ok=True)
    try:
        cmt, gsw = dati.carica_storico()
    except FileNotFoundError as e:
        sys.exit(str(e))
    par_t = dati.par_cmt(cmt).dropna(how="all")
    par_f = dati.par_gsw(gsw)
    print(f"Tesoro: {len(par_t)} giorni dal {par_t.index[0].date()} al {par_t.index[-1].date()}; "
          f"Fed: {len(par_f.dropna(how='all'))} giorni")

    # 1. premio del titolo on-the-run: punti del Tesoro meno par yield della curva Fed
    premio = an.differenza_bp(par_t, par_f)
    r = an.riepilogo(premio)
    mostra("Tesoro meno Fed (bp), per periodo", r.drop(index="intero", level=0), "premio_on_the_run")
    print("\nIntero periodo:\n", r.loc["intero"].to_string())
    confronto = pd.DataFrame({"t con 20 ritardi": an.riepilogo(premio, ritardi=20)["t_NW"],
                              "t con 250 ritardi": r["t_NW"]}).drop(index="intero", level=0)
    mostra("Premio: t con 20 e con 250 ritardi di Newey-West", confronto, "premio_t_ritardi")
    ac = pd.DataFrame({k: {c: d[c].dropna().autocorr(60) for c in d.columns} for k, d in an.periodi(premio).items()}).round(2)
    mostra("Premio: autocorrelazione a 60 giorni", ac, "premio_autocorr60")

    # 2. Nelson-Siegel sui punti del Tesoro
    print("\nAdattamento giornaliero della curva (2-3 minuti)...")
    parametri, residui = an.adatta_serie(par_t)
    parametri.to_csv(OUT / "ns_parametri.csv")
    residui.to_csv(OUT / "ns_residui_bp.csv")
    rmse = pd.DataFrame({k: np.sqrt((v**2).mean()) for k, v in an.periodi(residui).items()}).T.round(2)
    mostra("Residuo Nelson-Siegel sui punti del Tesoro: scarto quadratico medio (bp)", rmse, "ns_residui_rmse")
    print(f"Giorni con tau1 al limite superiore (10 anni): {(parametri.tau1 > 9.99).sum()} su {len(parametri)}; "
          f"al limite inferiore (0.3): {(parametri.tau1 < 0.31).sum()}")
    ns = an.curva_da_parametri(parametri, list(par_t.columns))
    d = an.differenza_bp(ns, par_f)
    mostra("Curva Nelson-Siegel (Tesoro) meno curva Fed: scarto quadratico medio (bp)",
           pd.DataFrame({k: np.sqrt((v**2).mean()) for k, v in an.periodi(d).items()}).T.round(2), "ns_contro_fed")

    # 3. PCA contro fattori
    # le variazioni si calcolano sulla tabella con i buchi (i 30 anni mancano dal 2002 al 2006) e poi si scartano
    # quelle con un dato mancante: altrimenti il primo giorno dopo il buco confronta due date lontane quattro anni
    completi = par_t.dropna()
    var, punteggi, vettori = an.pca(par_t.diff().dropna() * 100)
    fattori = an.fattori_dl(completi).diff()
    fattori = fattori[par_t.diff().notna().all(axis=1).reindex(fattori.index).fillna(False)]
    pca_t = pd.DataFrame({"varianza spiegata": var.round(3),
                          "R2 su 3 fattori NS": an.r2_su_fattori(punteggi, fattori).round(3).to_numpy()},
                         index=["PC1", "PC2", "PC3"])
    mostra(f"PCA delle variazioni giornaliere ({len(punteggi)} variazioni con tutte le scadenze)", pca_t, "pca")

    # 4. Diebold-Li: AR(1) contro passeggiata casuale
    righe = []
    for h in (1, 6, 12):
        rm, dm, n, cw = an.previsione_dl(completi, h, "2010-01-01")
        riga = rm.mean().round(1).to_dict()
        righe.append({"orizzonte (mesi)": h, "previsioni": n, "RMSE medio AR(1) (bp)": riga["AR(1)"],
                      "RMSE medio RW (bp)": riga["RW"], "t DM (>0: AR meglio)": dm, "t Clark-West": cw})
    mostra("Previsione fuori campione, test dal 2010 (media delle scadenze)", pd.DataFrame(righe).set_index("orizzonte (mesi)"), "previsione_dl")


if __name__ == "__main__":
    main()

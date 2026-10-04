"""Livello 3: mondo sintetico. Stampa e salva in output/ le tabelle di recupero, falsi positivi e potere.

Parametri fissati e dichiarati: 1500 giorni (circa 6 anni), emivita del mispricing 5 giorni,
rumore di misura 0.5 bp, 60 simulazioni per cella, soglia t di Newey-West = 2 (10 ritardi).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import mondo_sintetico as ms  # noqa: E402

OUT = ROOT / "output"
GIORNI, EMIVITA, RUMORE, SIM = 1500, 5.0, 0.5, 60
TAU_DL = 1.37  # lambda di Diebold-Li: 0.0609 per mese = 1.37 anni


def tabella_recupero():
    righe = []
    for nome, tau in (("tau libero", None), ("tau fisso = vero (2.0)", 2.0), ("tau fisso Diebold-Li (1.37)", TAU_DL)):
        r = pd.DataFrame([ms.recupero(GIORNI, RUMORE, 100 + i, tau) for i in range(SIM)]).mean()
        righe.append(pd.Series(r, name=nome))
    return pd.DataFrame(righe).round(3)


def tabella_falsi_positivi():
    righe = []
    for vero in ("ns", "svensson"):
        for nome, tau in (("tau libero", None), ("tau fisso 2.0", 2.0)):
            for costo in (0.0, 0.25):
                p, m = ms.potere(GIORNI, 0.0, EMIVITA, RUMORE, SIM, costo_bp=costo, vero=vero, seme=500, tau1=tau)
                righe.append({"curva vera": vero, "fit": nome, "costo (bp)": costo,
                              "frequenza t>2": round(p, 3), "guadagno medio (bp/giorno)": round(m, 4)})
    return pd.DataFrame(righe)


def tabella_potere():
    griglia = [0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    righe = []
    for s in griglia:
        riga = {"mispricing dev.std (bp)": s}
        for costo in (0.0, 0.25, 0.5):
            p, m = ms.potere(GIORNI, s, EMIVITA, RUMORE, SIM, costo_bp=costo, seme=900)
            riga[f"potere costo {costo}"] = round(p, 2)
            riga[f"guadagno costo {costo}"] = round(m, 4)
        righe.append(riga)
    return pd.DataFrame(righe)


def main():
    OUT.mkdir(exist_ok=True)
    rec, fp, pot = tabella_recupero(), tabella_falsi_positivi(), tabella_potere()
    for nome, t in (("recupero", rec), ("falsi_positivi", fp), ("potere", pot)):
        print(f"\n== {nome} ==")
        print(t.to_string())
        t.to_csv(OUT / f"sintetico_{nome}.csv")
    for costo in (0.0, 0.25, 0.5):
        ok = pot[pot[f"potere costo {costo}"] >= 0.8]["mispricing dev.std (bp)"]
        minimo = ok.min() if len(ok) else np.nan
        print(f"Mispricing minimo rilevabile (potere >= 0.8), costo {costo} bp: {minimo} bp")


if __name__ == "__main__":
    main()

"""Mondo sintetico (sezione 7 del README). Stampa e salva in output/ le tabelle di recupero, falsi positivi,
robustezza all'arrotondamento e potere.

Parametri: 1500 giorni (circa 6 anni), emivita del mispricing 5 giorni, rumore di misura 0.5 bp, 60 simulazioni per
cella nelle tabelle di recupero e falsi positivi, 100 nella tabella del potere; soglia t di Newey-West = 2 (10 ritardi).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import mondo_sintetico as ms  # noqa: E402

OUT = ROOT / "output"
GIORNI, EMIVITA, RUMORE, SIM, SIM_POTERE = 1500, 5.0, 0.5, 60, 100
TAU_DL = 1.37  # tau di Diebold-Li: lambda 0.0609 per mese, cioe' 1.37 anni


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


def tabella_arrotondamento():
    """I punti del Tesoro hanno due decimali (1 bp): il sintetico, senza arrotondamento, e' ottimista?"""
    righe = []
    for nome, tau in (("tau libero", None), ("tau fisso 2.0", 2.0)):
        for ris in (0.0, 1.0):
            fp, _ = ms.potere(GIORNI, 0.0, EMIVITA, RUMORE, SIM, costo_bp=0.25, seme=500, tau1=tau, risoluzione_bp=ris)
            pot, _ = ms.potere(GIORNI, 1.75, EMIVITA, RUMORE, SIM_POTERE, costo_bp=0.25, seme=900, tau1=tau,
                               risoluzione_bp=ris)
            fp0, _ = ms.potere(GIORNI, 0.0, EMIVITA, RUMORE, SIM, costo_bp=0.0, seme=500, tau1=tau, risoluzione_bp=ris)
            righe.append({"fit": nome, "arrotondamento (bp)": ris, "falsi positivi (costo 0)": round(fp0, 3),
                          "falsi positivi (costo 0.25)": round(fp, 3), "potere a 1.75 bp (costo 0.25)": round(pot, 2)})
    return pd.DataFrame(righe)


def tabella_potere(risoluzione_bp=0.0):
    """Potere (frazione di simulazioni con guadagno medio > 0 e t > 2) per mispricing, costo, fit ed esecuzione.
    Con risoluzione_bp = 1 i rendimenti sintetici sono arrotondati a 1 bp come i punti del Tesoro."""
    griglia = [0.1, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 3.25, 3.5, 4.0, 5.0]
    righe = []
    for s in griglia:
        riga = {"mispricing dev.std (bp)": s}
        for fit, tau in (("libero", None), ("tau fisso", 2.0)):
            for ritardo in (0, 1):
                for costo in (0.0, 0.25, 0.5):
                    if fit == "libero" and ritardo == 1:
                        continue
                    p, _ = ms.potere(GIORNI, s, EMIVITA, RUMORE, SIM_POTERE, costo_bp=costo, seme=900, tau1=tau,
                                     ritardo=ritardo, risoluzione_bp=risoluzione_bp)
                    riga[f"{fit}, {'giorno dopo' if ritardo else 'stesso giorno'}, costo {costo}"] = round(p, 2)
        righe.append(riga)
    return pd.DataFrame(righe).set_index("mispricing dev.std (bp)")


def main():
    OUT.mkdir(exist_ok=True)
    rec, fp, arr = tabella_recupero(), tabella_falsi_positivi(), tabella_arrotondamento()
    pot, pot_arr = tabella_potere(), tabella_potere(risoluzione_bp=1.0)
    for nome, t in (("recupero", rec), ("falsi_positivi", fp), ("arrotondamento", arr), ("potere", pot),
                    ("potere_arrotondato", pot_arr)):
        print(f"\n== {nome} ==")
        print(t.to_string())
        t.to_csv(OUT / f"sintetico_{nome}.csv")
    for titolo, t in (("senza arrotondamento", pot), ("con arrotondamento a 1 bp", pot_arr)):
        print(f"\nMispricing minimo rilevabile (potere >= 0.8), {titolo}: primo punto della griglia che lo raggiunge")
        for col in t.columns:
            ok = t.index[t[col] >= 0.8]
            print(f"  {col}: {ok.min() if len(ok) else 'nessuno'} bp")


if __name__ == "__main__":
    main()

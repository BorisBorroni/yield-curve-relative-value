"""Analisi di oggi: scarica la curva del Tesoro piu' recente, la adatta e la confronta con lo storico.

Uso: python scripts/analisi_corrente.py            (scarica i giorni dopo la fine dello storico)
     python scripts/analisi_corrente.py --senza-scarico   (riusa data/corrente/cmt_recente.csv)
I file scaricati vanno in data/corrente/ (non versionata); il risultato in output/.
Rispetta robots.txt; circa 2-3 richieste. Lo storico dei residui si ricalcola in 2-3 minuti la prima volta.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import analisi as an  # noqa: E402
from yield_curve_relative_value import corrente, dati  # noqa: E402

CORRENTE = ROOT / "data" / "corrente" / "cmt_recente.csv"
OUT = ROOT / "output"


def main():
    OUT.mkdir(exist_ok=True)
    CORRENTE.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmt, _ = dati.carica_storico()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if "--senza-scarico" not in sys.argv:
        try:
            recente = dati.scarica_cmt(dati.Scaricatore(), corrente.anni_da_scaricare(pd.Timestamp.today()))
        except (requests.RequestException, PermissionError) as e:
            sys.exit(f"Download non riuscito: {type(e).__name__} {str(e)[:200]}")
        recente.to_csv(CORRENTE)
    elif not CORRENTE.exists():
        sys.exit(f"{CORRENTE} non esiste: esegui lo script senza --senza-scarico")
    recente = pd.read_csv(CORRENTE, index_col=0, parse_dates=True)
    recente = recente[recente.index > pd.Timestamp(dati.FINE_STORICO)]
    par_storico = dati.par_cmt(cmt).dropna(how="all")
    par_recente = dati.par_cmt(recente).dropna(how="all")
    if par_recente.empty:
        sys.exit("Nessun dato dopo la fine dello storico.")

    f = OUT / "ns_residui_bp.csv"
    if f.exists():
        residui = pd.read_csv(f, index_col=0, parse_dates=True)
        residui.columns = [int(c) for c in residui.columns]
    else:
        print("Calcolo dei residui storici (2-3 minuti)...")
        residui = an.adatta_serie(par_storico)[1]
        residui.to_csv(f)

    t = corrente.tabella_oggi(par_storico, residui, par_recente)
    print(f"\nUltimo giorno disponibile: {t.attrs['ultimo_giorno'].date()} ({len(par_recente)} giorni dopo lo storico)")
    print("\n== Residui di Nelson-Siegel (bp) e z-score sui 60 giorni precedenti ==")
    print(t.to_string())
    ff = corrente.farfalle_oggi(par_storico, par_recente)
    print("\n== Farfalle ==")
    print(ff.to_string())
    t.to_csv(OUT / "corrente_residui.csv")
    ff.to_csv(OUT / "corrente_farfalle.csv")
    print("\nLettura: |z| alto non e' un segnale operativo. Nello storico i guadagni di questo segnale"
          " delle farfalle spariscono con costi fra 0.04 e 0.2 bp (vedi esegui_strategia.py).")


if __name__ == "__main__":
    main()

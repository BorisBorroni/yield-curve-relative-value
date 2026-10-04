"""Crea data/storico/ (curve del Tesoro e della Fed dal 2000 a fine 2025) se non esiste.

Rispetta robots.txt, una richiesta al secondo, ~30 richieste. Se i file ci sono gia' non fa nulla:
lo storico e' fisso e versionato. Per rigenerarlo, cancella la cartella data/storico/.
"""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import dati  # noqa: E402


def main():
    cartella = ROOT / "data" / "storico"
    if (cartella / "cmt.csv").exists() and (cartella / "gsw.csv").exists():
        print("Storico gia' presente in", cartella)
        return
    sc = dati.Scaricatore()
    try:
        cmt = dati.scarica_cmt(sc, range(2000, 2026))
        gsw = dati.scarica_gsw(sc)
    except (requests.RequestException, PermissionError) as e:
        sys.exit(f"Download non riuscito: {type(e).__name__} {str(e)[:200]}")
    cmt, gsw = dati.costruisci_storico(cmt, gsw)
    dati.salva_storico(cmt, gsw, cartella)
    print(f"Salvati cmt.csv ({len(cmt)} giorni) e gsw.csv ({len(gsw)} giorni) in {cartella}")


if __name__ == "__main__":
    main()

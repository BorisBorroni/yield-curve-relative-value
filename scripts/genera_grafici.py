"""Crea le due immagini del README in img/: premio del titolo nuovo e guadagno cumulato di una farfalla."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yield_curve_relative_value import analisi as an  # noqa: E402
from yield_curve_relative_value import dati, grafici  # noqa: E402
from yield_curve_relative_value import strategia as sg  # noqa: E402


def main():
    try:
        cmt, gsw = dati.carica_storico()
    except FileNotFoundError as e:
        sys.exit(str(e))
    (ROOT / "img").mkdir(exist_ok=True)
    par_t, par_f = dati.par_cmt(cmt).dropna(how="all"), dati.par_gsw(gsw)
    grafici.grafico_premio(an.differenza_bp(par_t, par_f)).savefig(ROOT / "img" / "premio_on_the_run.png", dpi=150)
    w = sg.pesi_farfalla((2, 5, 10), "dv01")
    x = sg.serie_farfalla(par_t, (2, 5, 10), w)
    cum = grafici.guadagni_cumulati(x, abs(w).sum())
    grafici.grafico_guadagni(cum, "Farfalla 2-5-10: guadagno cumulato della regola z-score").savefig(
        ROOT / "img" / "guadagni_farfalla.png", dpi=150)
    print("Salvati img/premio_on_the_run.png e img/guadagni_farfalla.png")


if __name__ == "__main__":
    main()

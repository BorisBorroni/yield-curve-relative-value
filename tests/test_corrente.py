import numpy as np
import pandas as pd
import pytest

from yield_curve_relative_value import corrente as co
from yield_curve_relative_value import curva as cv
from yield_curve_relative_value import dati
from yield_curve_relative_value import strategia as sg

N = np.array(dati.SCADENZE, float)


def test_anni_da_scaricare():
    assert list(co.anni_da_scaricare(pd.Timestamp("2027-03-01"))) == [2026, 2027]
    assert len(co.anni_da_scaricare(pd.Timestamp("2025-06-01"))) == 0


def test_percentile():
    assert co.percentile(2.0, [-3, -1, 0, 1, 2, 3]) == pytest.approx(100 * 4 / 6)
    assert co.percentile(-0.5, [np.nan, 0.1, 1.0]) == pytest.approx(50.0)


def test_zscore_esclude_il_giorno_corrente():
    x = np.random.default_rng(0).normal(size=80)
    z = sg.zscore(x, 30)
    assert np.isnan(z[:30]).all() and np.isfinite(z[30:]).all()
    y = x.copy()
    y[-1] = 100.0  # cambia solo l'ultimo giorno: i z precedenti non cambiano
    assert np.allclose(z[:-1], sg.zscore(y, 30)[:-1], equal_nan=True)
    manuale = (x[40] - x[10:40].mean()) / x[10:40].std(ddof=1)
    assert z[40] == pytest.approx(manuale)


def _panel(giorni, inizio, seme):
    rng = np.random.default_rng(seme)
    idx = pd.bdate_range(inizio, periods=giorni)
    y = cv.par_yield_vettore(N, cv.Parametri(4.0, -1.5, -1.0, 0.0, 2.0)) + rng.normal(0, 0.02, (giorni, len(N)))
    return pd.DataFrame(y, index=idx, columns=list(dati.SCADENZE))


def test_tabella_e_farfalle_oggi():
    storico = _panel(150, "2025-05-01", 1)
    recente = _panel(10, "2026-01-05", 2)
    recente.iloc[-1, 5] += 0.30  # il 10 anni salta di 30 bp l'ultimo giorno
    from yield_curve_relative_value import analisi as an
    residui = an.adatta_serie(storico)[1]
    t = co.tabella_oggi(storico, residui, recente, finestra=60)
    assert t.attrs["ultimo_giorno"] == recente.index[-1]
    assert t.loc[10, "z"] > 3 and t.loc[10, "percentile_|z|"] > 95
    ff = co.farfalle_oggi(storico, recente, finestra=60)
    assert ff.shape == (4, 3) and ff.loc[("2-5-10", "dv01"), "z"] != 0

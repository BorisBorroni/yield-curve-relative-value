import numpy as np
import pandas as pd
import pytest

from yield_curve_relative_value import curva as cv
from yield_curve_relative_value import strategia as sg


def test_pesi_dv01():
    assert sg.pesi_farfalla((2, 5, 10), "dv01") == pytest.approx([0.5, -1.0, 0.5])


@pytest.mark.parametrize("terna", [(2, 5, 10), (5, 10, 30)])
def test_pesi_fattori_annullano_livello_e_pendenza(terna):
    w = sg.pesi_farfalla(terna, "fattori", tau=1.37)
    carichi = cv.carichi(list(terna), 1.37)
    assert w[1] == -1.0
    assert w @ carichi[:, 0] == pytest.approx(0.0, abs=1e-12)  # livello
    assert w @ carichi[:, 1] == pytest.approx(0.0, abs=1e-12)  # pendenza
    assert w @ carichi[:, 2] != pytest.approx(0.0, abs=1e-3)   # resta la curvatura


def test_pesi_neutralita_sconosciuta():
    with pytest.raises(ValueError):
        sg.pesi_farfalla((2, 5, 10), "altro")


def test_serie_farfalla_valore_in_bp():
    par = pd.DataFrame({2: [2.0], 5: [3.5], 10: [6.0]}, index=pd.to_datetime(["2020-01-02"]))
    # 0.5*2.0 + 0.5*6.0 - 3.5 = +0.5 punti percentuali = +50 bp (ali sopra la pancia)
    assert sg.serie_farfalla(par, (2, 5, 10), [0.5, -1.0, 0.5]).iloc[0] == pytest.approx(50.0)


def test_posizioni_usano_solo_il_passato():
    x = np.random.default_rng(0).normal(size=300)
    base = sg.posizioni(x, finestra=30)
    y = x.copy()
    y[200:] += 10
    assert np.array_equal(base[:200], sg.posizioni(y, finestra=30)[:200])


def test_posizioni_scommettono_sul_rientro():
    x = np.zeros(100)
    x[:60] = np.random.default_rng(1).normal(0, 1, 60)
    x[80] = 10.0  # picco molto sopra la media -> +1 (guadagna se la serie scende)
    x[85] = -10.0
    pos = sg.posizioni(x, finestra=50, soglia=2.0)
    assert pos[80] == 1.0 and pos[85] == -1.0


def test_guadagno_di_un_picco_che_rientra():
    x = np.zeros(120)
    x[:70] = np.random.default_rng(2).normal(0, 1, 70)
    x[100] = 8.0
    g, pos = sg.guadagni_serie(x, x, finestra=60, soglia=3.0)
    assert pos[100] == 1.0
    assert g[100] == pytest.approx(8.0)  # la serie torna a 0 il giorno dopo


def test_costo_proporzionale_ai_pesi_e_ai_cambi():
    x = np.zeros(120)
    x[:70] = np.random.default_rng(3).normal(0, 1, 70)
    x[100] = 8.0
    g0, _ = sg.guadagni_serie(x, x, somma_pesi=2.0, finestra=60, soglia=3.0, costo_bp=0.0)
    g1, _ = sg.guadagni_serie(x, x, somma_pesi=2.0, finestra=60, soglia=3.0, costo_bp=0.25)
    # entrata nel giorno 100 (cambio di 1) e uscita il 101 (cambio di 1): 0.25 * 2 per cambio
    assert g0.sum() - g1.sum() == pytest.approx(2 * 0.25 * 2.0)


def test_emivita_di_un_ar1():
    rng = np.random.default_rng(4)
    phi = 0.5 ** (1 / 10)
    x = np.zeros(3000)
    for t in range(1, 3000):
        x[t] = phi * x[t - 1] + rng.normal()
    h = sg.emivita(x, 2000)
    assert h[2500] == pytest.approx(10, rel=0.4)
    camminata = np.cumsum(rng.normal(size=300))
    assert np.isinf(sg.emivita(camminata, 250)[:250]).all()


def test_ritardo_sposta_di_un_giorno_l_esecuzione():
    x = np.zeros(130)
    x[:70] = np.random.default_rng(2).normal(0, 1, 70)
    x[100], x[101], x[102] = 8.0, 4.0, 0.0
    g0, pos = sg.guadagni_serie(x, x, finestra=60, soglia=3.0)
    g1, _ = sg.guadagni_serie(x, x, finestra=60, soglia=3.0, ritardo=1)
    assert pos[100] == 1.0 and len(g1) == len(g0) - 1
    # posizione presa al giorno 100 con x=8, aperta al giorno 101 (x=4): guadagna x_101 - x_102 = 4
    assert g1[100] == pytest.approx(4.0)


def test_zscore_con_finestra_costante_e_nan():
    z = sg.zscore(np.concatenate([np.ones(70), [5.0]]), finestra=60)
    assert np.isnan(z[:60]).all() and np.isnan(z[-1])


def test_pesi_farfalla_scadenze_non_crescenti():
    with pytest.raises(ValueError):
        sg.pesi_farfalla((5, 5, 10), "fattori")

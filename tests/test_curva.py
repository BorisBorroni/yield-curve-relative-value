from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_curve_relative_value import curva as cv
from yield_curve_relative_value import obbligazioni as ob

CAMPIONE = Path(__file__).parent / "dati" / "gsw_campione.csv"
P = cv.Parametri(4.5, -2.0, -1.5, 1.0, 1.7, 9.0)
REG = date(2026, 10, 3)


def parametri_riga(r):
    """Parametri GSW di una riga; nel periodo Nelson-Siegel beta3 = 0 e tau2 e' un codice (-999.99)."""
    return cv.Parametri(r.BETA0, r.BETA1, r.BETA2, r.BETA3, r.TAU1, r.TAU2 if r.BETA3 != 0 else 5.0)


@pytest.fixture(scope="module")
def gsw():
    return pd.read_csv(CAMPIONE, index_col=0, parse_dates=True)


def test_riproduce_zero_forward_e_par_yield_della_fed(gsw):
    # i numeri pubblicati dalla Fed hanno 4 decimali: differenza massima ammessa 1e-4 punti
    n = np.arange(1, 31)
    for _, r in gsw.iterrows():
        p = parametri_riga(r)
        zero = np.array([r[f"SVENY{k:02d}"] for k in n])
        fwd = np.array([r[f"SVENF{k:02d}"] for k in n])
        ok = ~np.isnan(zero)
        assert np.allclose(cv.rendimento_zero(n, p)[ok], zero[ok], atol=1e-4)
        ok = ~np.isnan(fwd)
        assert np.allclose(cv.forward(n, p)[ok], fwd[ok], atol=1e-4)
        for k in n:
            par = r[f"SVENPY{k:02d}"]
            if not np.isnan(par):
                assert cv.par_yield(k, p) == pytest.approx(par, abs=1e-4)


def test_limiti_della_curva():
    assert cv.rendimento_zero(1e-9, P)[0] == pytest.approx(P.beta0 + P.beta1, abs=1e-6)
    assert cv.rendimento_zero(1e6, P)[0] == pytest.approx(P.beta0, abs=1e-3)


def test_ns_e_svensson_con_beta3_nullo():
    ns = cv.Parametri(4.0, -1.0, -2.0, 0.0, 2.0)
    sv = cv.Parametri(4.0, -1.0, -2.0, 0.0, 2.0, 7.0)
    n = np.array([0.5, 2, 10])
    assert np.allclose(cv.rendimento_zero(n, ns), cv.rendimento_zero(n, sv))
    assert ns.ns and not P.ns


def test_forward_e_derivata_di_y_per_n():
    # f(n) = d/dn [ n y(n) ]
    n, h = 4.0, 1e-5
    num = ((n + h) * cv.rendimento_zero(n + h, P) - (n - h) * cv.rendimento_zero(n - h, P)) / (2 * h)
    assert cv.forward(n, P)[0] == pytest.approx(num[0], abs=1e-6)


def test_fattore_di_sconto():
    assert cv.fattore_sconto(0.0, P) == pytest.approx(1.0)
    n = np.array([1.0, 5.0, 20.0])
    assert np.all(np.diff(cv.fattore_sconto(n, P)) < 0)


def test_par_yield_scadenza_non_semestrale():
    with pytest.raises(ValueError):
        cv.par_yield(2.3, P)


def test_titolo_a_cedola_par_yield_vale_100():
    # su una data cedola, la cedola pari al par yield dà prezzo 100. Non esattamente: il par yield
    # usa mezzi anni esatti, il prezzo usa i giorni reali / 365.25 (scarto di circa 0.004 su 100)
    reg, scad = date(2026, 11, 15), date(2031, 11, 15)
    c = cv.par_yield(5, P)
    assert cv.prezzo_sporco_modello(c, scad, reg, P) == pytest.approx(100.0, abs=0.01)


def test_pesi_vicini_all_inverso_della_duration():
    # titoli alla pari: 1/duration pesa gli errori di prezzo come errori di rendimento
    titoli = [cv.Titolo(4.0, date(2026 + k, 11, 15), 100.0) for k in (1, 3, 10)]
    reg = date(2026, 11, 15)
    pesi = cv._pesi(titoli, reg)
    d = [ob.duration_modificata(4.0, 4.0, t.scadenza, reg) for t in titoli]
    assert np.allclose(pesi, 1 / np.array(d), rtol=1e-6)


def _titoli_sintetici(p, scadenze, cedole):
    return [cv.Titolo(c, date(2026 + int(s), 11, 15), cv.prezzo_sporco_modello(c, date(2026 + int(s), 11, 15), date(2026, 11, 15), p))
            for s, c in zip(scadenze, cedole, strict=True)]


def test_calibrazione_prezzi_recupera_nelson_siegel():
    vero = cv.Parametri(4.8, -2.2, -1.4, 0.0, 2.3)
    scad = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    titoli = _titoli_sintetici(vero, scad, [3.0, 2.5, 3.5, 4.0, 4.5, 4.0, 5.0, 4.5, 4.0, 4.5])
    stima = cv.calibra_prezzi(titoli, date(2026, 11, 15))
    assert stima.beta0 == pytest.approx(vero.beta0, abs=1e-4)
    assert stima.beta1 == pytest.approx(vero.beta1, abs=1e-4)
    assert stima.beta2 == pytest.approx(vero.beta2, abs=1e-4)
    assert stima.tau1 == pytest.approx(vero.tau1, abs=1e-3)


def test_calibrazione_con_lambda_fisso_diebold_li():
    vero = cv.Parametri(4.8, -2.2, -1.4, 0.0, 1.37)
    scad = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    titoli = _titoli_sintetici(vero, scad, [3.0] * 10)
    stima = cv.calibra_prezzi(titoli, date(2026, 11, 15), tau1=1.37)
    assert stima.tau1 == 1.37
    assert stima.beta0 == pytest.approx(vero.beta0, abs=1e-4)


def test_adatta_rendimenti_recupera_parametri():
    vero = cv.Parametri(4.0, -1.5, 2.0, 0.0, 3.0)
    n = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = cv.rendimento_zero(n, vero)
    fisso = cv.adatta_rendimenti(n, y, tau1=3.0)
    assert (fisso.beta0, fisso.beta1, fisso.beta2) == pytest.approx((4.0, -1.5, 2.0), abs=1e-9)
    libero = cv.adatta_rendimenti(n, y, griglia=np.linspace(0.5, 8, 151))
    assert libero.tau1 == pytest.approx(3.0, abs=1e-9)
    assert libero.beta0 == pytest.approx(4.0, abs=1e-9)


def test_condizionamento_cresce_con_tau_grande():
    n = np.array([1, 2, 5, 10, 30.0])
    assert cv.condizionamento(n, 1.0) < cv.condizionamento(n, 15.0)
    # con scadenze brevi e tau grande i carichi sono quasi collineari
    assert cv.condizionamento(np.array([1, 2, 3.0]), 15.0) > 1e3


def test_par_yield_vettore_come_lo_scalare():
    n = np.array([1.0, 2.5, 10.0, 30.0])
    assert np.allclose(cv.par_yield_vettore(n, P), [cv.par_yield(k, P) for k in n])
    with pytest.raises(ValueError):
        cv.par_yield_vettore(np.array([2.3]), P)


def test_adatta_par_yield_recupera_i_parametri():
    vero = cv.Parametri(4.5, -2.0, -1.0, 0.0, 2.0)
    n = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
    stima = cv.adatta_par_yield(n, cv.par_yield_vettore(n, vero))
    assert (stima.beta0, stima.beta1, stima.beta2, stima.tau1) == pytest.approx((4.5, -2.0, -1.0, 2.0), abs=1e-5)


def test_adatta_par_yield_ignora_i_mancanti():
    vero = cv.Parametri(4.5, -2.0, -1.0, 0.0, 2.0)
    n = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
    y = cv.par_yield_vettore(n, vero)
    y[-1] = np.nan
    assert cv.adatta_par_yield(n, y).beta0 == pytest.approx(4.5, abs=1e-4)
    with pytest.raises(ValueError):
        cv.adatta_par_yield(n[:3], y[:3])


def test_residui_panel_con_tau_fisso_e_regressione():
    n = np.array([1, 2, 5, 10, 30.0])
    y = np.tile(cv.rendimento_zero(n, cv.Parametri(4.0, -1.0, 1.0, 0.0, 1.37)), (3, 1))
    beta, tau, res = cv.residui_panel(y, n, tau1=1.37)
    assert np.allclose(beta, [4.0, -1.0, 1.0]) and np.allclose(tau, 1.37) and np.abs(res).max() < 1e-9


def test_parametri_con_tau_non_positivo():
    with pytest.raises(ValueError):
        cv.Parametri(4.0, -1.0, 1.0, 0.0, tau1=0.0)


def test_residui_panel_rifiuta_i_valori_mancanti():
    y = np.array([[4.0, np.nan, 4.2, 4.3]])
    with pytest.raises(ValueError):
        cv.residui_panel(y, [1, 2, 5, 10])

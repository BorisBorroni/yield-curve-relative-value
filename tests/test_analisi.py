import numpy as np
import pandas as pd
import pytest

from yield_curve_relative_value import analisi as an
from yield_curve_relative_value import curva as cv

N = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])


def panel_esatto(giorni=6):
    idx = pd.bdate_range("2021-12-01", periods=giorni)
    righe = [cv.par_yield_vettore(N, cv.Parametri(4.0 + 0.1 * i, -2.0, -1.0, 0.0, 2.0)) for i in range(giorni)]
    return pd.DataFrame(righe, index=idx, columns=N)


def test_adatta_serie_su_dati_esatti():
    par, res = an.adatta_serie(panel_esatto())
    assert np.abs(res.to_numpy()).max() < 1e-3  # bp
    assert par["beta0"].iloc[-1] == pytest.approx(4.5, abs=1e-4)


def test_curva_da_parametri_ritrova_i_dati():
    y = panel_esatto(3)
    par, _ = an.adatta_serie(y)
    assert np.allclose(an.curva_da_parametri(par, list(N)).to_numpy(), y.to_numpy(), atol=1e-5)


def test_differenza_bp_e_giorni_comuni():
    a = panel_esatto(4)
    b = a.iloc[1:] + 0.01
    d = an.differenza_bp(a, b)
    assert d.shape == (3, 8) and np.allclose(d.to_numpy(), -1.0)


def test_periodi_separano_al_cambio_di_metodo():
    idx = pd.to_datetime(["2021-12-03", "2021-12-06", "2021-12-07"])
    p = an.periodi(pd.DataFrame({1: [1, 2, 3.0]}, index=idx))
    assert len(p["fino a dic 2021"]) == 1 and len(p["da dic 2021"]) == 2 and len(p["intero"]) == 3


def test_riepilogo_media_e_t():
    idx = pd.bdate_range("2019-01-01", periods=400)
    x = pd.DataFrame({10: np.random.default_rng(0).normal(-5.0, 1.0, 400)}, index=idx)
    r = an.riepilogo(x).loc[("intero", 10)]
    assert r["media"] == pytest.approx(-5.0, abs=0.2) and r["t_NW"] < -20 and r["giorni"] == 400


def test_pca_su_dati_di_rango_uno():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(500, 1))
    x = pd.DataFrame(f @ np.array([[1.0, 2.0, 3.0]]), index=pd.bdate_range("2020-01-01", periods=500))
    var, punteggi, vet = an.pca(x, componenti=2)
    assert var[0] == pytest.approx(1.0, abs=1e-9)
    assert abs(vet.iloc[:, 0] @ np.array([1.0, 2.0, 3.0])) == pytest.approx(np.sqrt(14), rel=1e-9)
    assert punteggi.shape == (500, 2)


def test_r2_su_fattori_vale_uno_se_la_componente_e_combinazione_dei_fattori():
    rng = np.random.default_rng(2)
    f = pd.DataFrame(rng.normal(size=(100, 3)), index=pd.bdate_range("2020-01-01", periods=100), columns=list("abc"))
    pc = pd.DataFrame({0: f["a"] + 2 * f["b"], 1: pd.Series(rng.normal(size=100), index=f.index)})
    r2 = an.r2_su_fattori(pc, f)
    assert r2[0] == pytest.approx(1.0) and r2[1] < 0.2


def _panel_mean_reverting(anni=20, seed=3):
    """Fattori con forte ritorno alla media (autocorrelazione mensile 0.5): l'AR(1) deve battere la passeggiata casuale."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=252 * anni)
    phi = 0.5 ** (1 / 21)
    media = np.array([4.0, -1.0, 1.0])
    b = np.empty((len(idx), 3))
    b[0] = media
    for t in range(1, len(idx)):
        b[t] = media + phi * (b[t - 1] - media) + 0.3 * np.sqrt(1 - phi**2) * rng.standard_normal(3)
    y = b @ cv.carichi(N, an.TAU_DL).T
    return pd.DataFrame(y, index=idx, columns=N)


@pytest.mark.slow
def test_previsione_dl_ar1_batte_la_passeggiata_con_fattori_che_tornano_alla_media():
    rmse, dm, n = an.previsione_dl(_panel_mean_reverting(), 1, "2008-01-01")
    assert (rmse["AR(1)"] < rmse["RW"]).all() and dm > 2 and n > 100

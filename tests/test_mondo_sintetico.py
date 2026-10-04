import numpy as np
import pytest

from yield_curve_relative_value import mondo_sintetico as ms


def test_ou_dev_std_ed_emivita():
    rng = np.random.default_rng(0)
    m = ms.genera_ou(40000, 1, 2.0, 5.0, rng)[:, 0]
    assert m.std() == pytest.approx(2.0, rel=0.05)
    assert np.corrcoef(m[:-5], m[5:])[0, 1] == pytest.approx(0.5, abs=0.03)  # dopo un'emivita: 0.5


def test_forma_e_riproducibilita():
    y1, mis1, ru1, b1 = ms.genera_panel(50, 1.0, 5.0, 0.5, np.random.default_rng(7))
    y2, *_ = ms.genera_panel(50, 1.0, 5.0, 0.5, np.random.default_rng(7))
    assert y1.shape == (50, len(ms.SCADENZE)) and b1.shape == (50, 3)
    assert np.array_equal(y1, y2)


def test_senza_mispricing_ne_rumore_la_curva_e_ns_esatta():
    y, _, _, _ = ms.genera_panel(30, 0.0, 5.0, 0.0, np.random.default_rng(1))
    from yield_curve_relative_value import curva as cv
    _, _, res = cv.residui_panel(y, ms.SCADENZE)
    assert np.abs(res).max() < 1e-3  # griglia di tau: scarto minimo


def test_panel_con_curva_non_valida():
    with pytest.raises(ValueError):
        ms.genera_panel(10, 0.0, 5.0, 0.0, np.random.default_rng(0), vero="altro")


def test_il_segnale_usa_solo_il_passato():
    rng = np.random.default_rng(2)
    r = rng.normal(size=(200, 3))
    base = ms.guadagni(r, r, finestra=30)
    r2 = r.copy()
    r2[150:] += 5.0  # cambio il futuro: i guadagni fino al giorno 148 non devono cambiare
    cambiato = ms.guadagni(r2, r, finestra=30)
    assert np.allclose(base[:149], cambiato[:149])


def test_costo_riduce_il_guadagno():
    r = np.random.default_rng(3).normal(size=(300, 4))
    assert ms.guadagni(r, r, costo_bp=0.5).mean() < ms.guadagni(r, r, costo_bp=0.0).mean()


@pytest.mark.slow
def test_senza_mispricing_con_tau_fisso_non_ci_sono_falsi_positivi_sistematici():
    p, media = ms.potere(1500, 0.0, 5.0, 0.5, 30, tau1=ms.TAU_VERO)
    assert p < 0.2 and abs(media) < 0.005


@pytest.mark.slow
def test_il_potere_cresce_con_il_mispricing():
    basso = ms.potere(1500, 0.5, 5.0, 0.5, 20, costo_bp=0.25)[0]
    alto = ms.potere(1500, 4.0, 5.0, 0.5, 20, costo_bp=0.25)[0]
    assert alto > 0.9 and basso < 0.5


@pytest.mark.slow
def test_i_parametri_si_stimano_peggio_della_curva():
    r = ms.recupero(1500, 0.5, 11)
    assert r["rmse_curva_bp"] < 0.5
    assert r["rmse_tau"] > 0.3 or r["rmse_beta2"] > 0.1

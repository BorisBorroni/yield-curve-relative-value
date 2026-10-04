from datetime import date

import pytest

from yield_curve_relative_value import obbligazioni as ob

SCAD = date(2030, 5, 15)


def test_date_cedole_fine_mese():
    # scadenza a fine febbraio: le cedole cadono a fine agosto e fine febbraio
    prec, future = ob.date_cedole(date(2030, 2, 28), date(2026, 10, 3))
    assert prec == date(2026, 8, 31)
    assert future[0] == date(2027, 2, 28)
    assert future[-1] == date(2030, 2, 28)
    assert len(future) == 7


def test_date_cedole_giorno_normale():
    prec, future = ob.date_cedole(SCAD, date(2026, 10, 3))
    assert prec == date(2026, 5, 15)
    assert future[0] == date(2026, 11, 15)
    assert len(future) == 8


def test_regolamento_sulla_data_cedola():
    prec, future = ob.date_cedole(SCAD, date(2026, 11, 15))
    assert prec == date(2026, 11, 15)
    assert future[0] == date(2027, 5, 15)
    assert ob.rateo(4.0, SCAD, date(2026, 11, 15)) == 0.0


def test_regolamento_dopo_scadenza():
    with pytest.raises(ValueError):
        ob.date_cedole(SCAD, SCAD)


def test_rateo_acts():
    # 140 giorni su 184 del periodo 15/05-15/11
    r = ob.rateo(4.0, SCAD, date(2026, 10, 2))
    assert r == pytest.approx(2.0 * 140 / 184)


def test_titolo_alla_pari_su_data_cedola():
    reg = date(2026, 11, 15)
    for c in (1.0, 4.5, 8.0):
        assert ob.prezzo_pulito(c, c, SCAD, reg) == pytest.approx(100.0, abs=1e-10)


def test_formula_chiusa_rendita():
    # su data cedola il prezzo e' la rendita: c/2 * a(n, y/2) + 100 v^n
    reg, c, y, n = date(2026, 11, 15), 3.0, 5.0, 7
    i = y / 200
    attesa = c / 2 * (1 - (1 + i) ** -n) / i + 100 * (1 + i) ** -n
    assert ob.prezzo_sporco(y, c, SCAD, reg) == pytest.approx(attesa, abs=1e-10)


def test_zero_cedola_formula():
    reg = date(2026, 11, 15)
    c, y = 0.0, 4.0
    atteso = 100 / (1 + y / 200) ** 7
    assert ob.prezzo_sporco(y, c, SCAD, reg) == pytest.approx(atteso, abs=1e-10)


def test_prezzo_decresce_con_rendimento():
    reg = date(2026, 10, 3)
    prezzi = [ob.prezzo_pulito(y, 4.0, SCAD, reg) for y in (2, 3, 4, 5, 6)]
    assert prezzi == sorted(prezzi, reverse=True)


@pytest.mark.parametrize("c,y", [(0.0, 3.0), (2.5, 4.2), (6.0, 1.0), (4.0, 4.0)])
def test_rendimento_inverte_prezzo(c, y):
    reg = date(2026, 10, 3)
    p = ob.prezzo_pulito(y, c, SCAD, reg)
    assert ob.rendimento(p, c, SCAD, reg) == pytest.approx(y, abs=1e-9)
    s = ob.prezzo_sporco(y, c, SCAD, reg)
    assert ob.rendimento(s, c, SCAD, reg, pulito=False) == pytest.approx(y, abs=1e-9)


def test_duration_zero_coupon():
    # senza cedola la duration di Macaulay e' il tempo alla scadenza
    reg = date(2026, 11, 15)
    assert ob.duration_macaulay(4.0, 0.0, SCAD, reg) == pytest.approx(3.5)


def test_dv01_e_convessita_contro_differenze_finite():
    reg, c, y = date(2026, 10, 3), 4.0, 4.5
    h = 0.01  # 1 punto base in %
    p_su = ob.prezzo_sporco(y + h, c, SCAD, reg)
    p_giu = ob.prezzo_sporco(y - h, c, SCAD, reg)
    p0 = ob.prezzo_sporco(y, c, SCAD, reg)
    assert ob.dv01(y, c, SCAD, reg) == pytest.approx((p_giu - p_su) / 2, rel=1e-5)
    d2 = (p_su - 2 * p0 + p_giu) / (h / 100) ** 2
    assert ob.convessita(y, c, SCAD, reg) == pytest.approx(d2 / p0, rel=1e-4)


def test_duration_modificata_e_derivata_logaritmica():
    reg, c, y = date(2026, 10, 3), 4.0, 4.5
    h = 0.001
    p_su = ob.prezzo_sporco(y + h, c, SCAD, reg)
    p_giu = ob.prezzo_sporco(y - h, c, SCAD, reg)
    p0 = ob.prezzo_sporco(y, c, SCAD, reg)
    numerica = -(p_su - p_giu) / (2 * h / 100) / p0
    assert ob.duration_modificata(y, c, SCAD, reg) == pytest.approx(numerica, rel=1e-6)


def test_flussi():
    f = ob.flussi(4.0, SCAD, date(2026, 10, 3))
    assert len(f) == 8
    assert f[0] == (date(2026, 11, 15), 2.0)
    assert f[-1] == (SCAD, 102.0)

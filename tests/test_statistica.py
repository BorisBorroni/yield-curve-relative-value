import numpy as np
import pytest

from yield_curve_relative_value import statistica as st


def test_newey_west_senza_ritardi_e_il_t_ordinario():
    x = np.random.default_rng(0).normal(0.3, 1.0, 500)
    media, t = st.newey_west(x, 0)
    assert media == pytest.approx(x.mean())
    assert t == pytest.approx(x.mean() / (x.std() / np.sqrt(len(x))))


def test_newey_west_riduce_t_con_autocorrelazione_positiva():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i] + 0.05
    assert abs(st.newey_west(x, 20)[1]) < abs(st.newey_west(x, 0)[1])

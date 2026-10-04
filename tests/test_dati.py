import pandas as pd
import pytest

from yield_curve_relative_value import dati


def test_etichette_e_anni():
    assert dati.etichetta("1 Mo") == "1Mo"
    assert dati.etichetta("1.5 Month") == "1.5Mo"
    assert dati.anni_da_etichetta("6Mo") == 0.5
    assert dati.anni_da_etichetta("10Yr") == 10
    with pytest.raises(ValueError):
        dati.anni_da_etichetta("abc")


CMT = "Date,3 Mo,10 Yr,30 Yr\n01/03/2000,5.5,6.5,\n01/04/2000,5.4,6.4,6.6\n"


def test_leggi_cmt_ordina_e_converte():
    df = dati.leggi_cmt(CMT)
    assert list(df.columns) == ["3Mo", "10Yr", "30Yr"]
    assert df.index.is_monotonic_increasing and df.index[0] == pd.Timestamp("2000-01-03")
    assert pd.isna(df.loc["2000-01-03", "30Yr"]) and df.loc["2000-01-04", "10Yr"] == 6.4


def test_leggi_gsw_salta_le_righe_iniziali():
    testo = 'Descrizione\nNote varie\n\n"Date",SVENY01,SVENPY01\n2000-01-03,6.1,6.2\n2000-01-04,-999.99,\n'
    df = dati.leggi_gsw(testo)
    assert df.shape == (2, 2) and df.loc["2000-01-03", "SVENPY01"] == 6.2


class Risposta:
    def __init__(self, codice, testo=""):
        self.status_code, self.text = codice, testo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class SessioneFinta:
    def __init__(self, robots):
        self.headers, self.robots, self.richieste = {}, robots, []

    def get(self, url, **kw):
        self.richieste.append(url)
        return self.robots if url.endswith("/robots.txt") else Risposta(200, "contenuto")


@pytest.mark.parametrize("robots,atteso", [
    (Risposta(404), True),
    (Risposta(403), False),
    (Risposta(200, "User-agent: *\nDisallow: /privato/"), True),
    (Risposta(200, "User-agent: *\nDisallow: /"), False),
])
def test_robots(robots, atteso):
    sc = dati.Scaricatore(SessioneFinta(robots), pausa=0)
    assert sc.consentito("https://esempio.org/dati/file.csv") is atteso


def test_se_robots_vieta_non_si_scarica():
    s = SessioneFinta(Risposta(200, "User-agent: *\nDisallow: /"))
    sc = dati.Scaricatore(s, pausa=0)
    with pytest.raises(PermissionError):
        sc.testo("https://esempio.org/x.csv")
    assert s.richieste == ["https://esempio.org/robots.txt"]  # nessuna richiesta del file


def test_storico_andata_e_ritorno(tmp_path):
    cmt = dati.leggi_cmt(CMT)
    gsw = pd.DataFrame({c: [1.0, 2.0] for c in dati.colonne_gsw()}, index=cmt.index)
    c2, g2 = dati.costruisci_storico(cmt, gsw)
    dati.salva_storico(c2, g2, tmp_path)
    c3, g3 = dati.carica_storico(tmp_path)
    assert c3.shape == c2.shape and g3.shape == g2.shape
    assert (c3.fillna(0) - c2.fillna(0)).abs().max().max() < 1e-9


def test_storico_taglia_dopo_la_fine():
    idx = pd.to_datetime(["2025-12-30", "2026-01-02"])
    cmt = pd.DataFrame({"10Yr": [4.0, 4.1]}, index=idx)
    gsw = pd.DataFrame({c: [1.0, 1.0] for c in dati.colonne_gsw()}, index=idx)
    c, g = dati.costruisci_storico(cmt, gsw)
    assert len(c) == 1 and len(g) == 1


def test_storico_mancante(tmp_path):
    with pytest.raises(FileNotFoundError, match="scarica_storico"):
        dati.carica_storico(tmp_path)


def test_selezione_scadenze():
    cmt = dati.leggi_cmt("Date,1 Yr,2 Yr\n01/03/2000,1.0,2.0\n")
    assert list(dati.par_cmt(cmt, (1, 2)).columns) == [1, 2]

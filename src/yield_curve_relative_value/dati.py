"""Lettura, download e storico dei dati (solo fonti gratuite).

Fonti:
- punti a scadenza costante (par yield) del Tesoro USA, un CSV per anno;
- curva Fed di Gurkaynak-Sack-Wright (GSW), un unico CSV;

Prima di ogni richiesta si legge il robots.txt dell'host: se vieta, non si scarica.
Il progetto tiene in data/storico/ uno storico fisso e piccolo (fino alla fine del 2025);
i dati piu' recenti si scaricano a ogni esecuzione in data/corrente/ (non versionata).
"""
import io
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd

UA = "yield-curve-relative-value/0.1 (progetto universitario)"
PAUSA = 1.0
URL_CMT = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{a}/all?type=daily_treasury_yield_curve"
           "&field_tdr_date_value={a}&page&_format=csv")
URL_GSW = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

# scadenze (anni) usate nell'analisi, e fine dello storico versionato
SCADENZE = (1, 2, 3, 5, 7, 10, 20, 30)
FINE_STORICO = "2025-12-31"
INIZIO_STORICO = "2000-01-01"
ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- lettura
def etichetta(nome: str) -> str:
    """'1 Mo' -> '1Mo', '1.5 Month' -> '1.5Mo', '10 Yr' -> '10Yr'."""
    return nome.replace("Month", "Mo").replace(" ", "")


def anni_da_etichetta(et: str) -> float:
    """Scadenza in anni da un'etichetta come '6Mo' o '10Yr'."""
    m = re.fullmatch(r"([\d.]+)(Mo|Yr)", et)
    if not m:
        raise ValueError(f"etichetta di scadenza non riconosciuta: {et}")
    return float(m.group(1)) / (12 if m.group(2) == "Mo" else 1)


def leggi_cmt(testo: str) -> pd.DataFrame:
    """CSV annuale del Tesoro -> tabella con indice data (crescente) e colonne '1Mo', ..., '30Yr'."""
    df = pd.read_csv(io.StringIO(testo), dtype=str)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    df.columns = [etichetta(c) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def leggi_gsw(testo: str) -> pd.DataFrame:
    """CSV della Fed -> tabella con indice data. L'intestazione e' la riga che inizia con 'Date'."""
    righe = testo.splitlines()
    i = next(k for k, r in enumerate(righe) if r.strip('"').startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(righe[i:])), parse_dates=["Date"], index_col="Date")
    return df.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------- download
class Scaricatore:
    """Sessione HTTP che rispetta robots.txt (404 = nessuna regola; 401/403 = vietato) e fa una pausa fra le richieste."""

    def __init__(self, sessione=None, pausa: float = PAUSA):
        if sessione is None:
            import requests

            sessione = requests.Session()
        self.s = sessione
        self.s.headers["User-Agent"] = UA
        self.pausa = pausa
        self._robots: dict[str, RobotFileParser] = {}

    def consentito(self, url: str) -> bool:
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            rp = RobotFileParser()
            r = self.s.get(host + "/robots.txt", timeout=30)
            if r.status_code == 404:
                rp.allow_all = True
            elif r.status_code in (401, 403):
                rp.disallow_all = True
            else:
                rp.parse(r.text.splitlines())
            self._robots[host] = rp
            time.sleep(self.pausa)
        return self._robots[host].can_fetch(UA, url)

    def testo(self, url: str, **kw) -> str:
        if not self.consentito(url):
            raise PermissionError(f"robots.txt vieta l'accesso a {url}")
        time.sleep(self.pausa)
        r = self.s.get(url, timeout=180, **kw)
        r.raise_for_status()
        return r.text


def scarica_cmt(sc: Scaricatore, anni) -> pd.DataFrame:
    """Scarica i file annuali e li unisce in una tabella (colonne = unione delle scadenze)."""
    return pd.concat([leggi_cmt(sc.testo(URL_CMT.format(a=a))) for a in anni]).sort_index()


def scarica_gsw(sc: Scaricatore) -> pd.DataFrame:
    return leggi_gsw(sc.testo(URL_GSW))


# ------------------------------------------------------------------ storico
def colonne_gsw():
    return [f"SVENPY{k:02d}" for k in SCADENZE] + [f"SVENY{k:02d}" for k in SCADENZE]


def costruisci_storico(cmt: pd.DataFrame, gsw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Taglia i dati grezzi sul periodo dello storico e tiene solo le colonne usate."""
    cmt = cmt.loc[INIZIO_STORICO:FINE_STORICO].dropna(how="all")
    ordine = sorted(cmt.columns, key=anni_da_etichetta)
    gsw = gsw.loc[INIZIO_STORICO:FINE_STORICO, colonne_gsw()].dropna(how="all")
    return cmt[ordine], gsw


def salva_storico(cmt, gsw, cartella: Path | None = None):
    cartella = cartella or ROOT / "data" / "storico"
    cartella.mkdir(parents=True, exist_ok=True)
    cmt.to_csv(cartella / "cmt.csv", float_format="%.2f")
    gsw.to_csv(cartella / "gsw.csv", float_format="%.4f")


def carica_storico(cartella: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legge lo storico versionato. Errore chiaro se manca (si crea con scripts/scarica_storico.py)."""
    cartella = cartella or ROOT / "data" / "storico"
    try:
        cmt = pd.read_csv(cartella / "cmt.csv", index_col=0, parse_dates=True)
        gsw = pd.read_csv(cartella / "gsw.csv", index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e.filename} non trovato: esegui scripts/scarica_storico.py") from e
    return cmt, gsw


def par_cmt(cmt: pd.DataFrame, scadenze=SCADENZE) -> pd.DataFrame:
    """Colonne del Tesoro alle scadenze indicate (anni interi), con colonne numeriche = scadenza."""
    out = pd.DataFrame({k: cmt[f"{k}Yr"] for k in scadenze})
    return out.astype(float)


def par_gsw(gsw: pd.DataFrame, scadenze=SCADENZE, zero: bool = False) -> pd.DataFrame:
    pref = "SVENY" if zero else "SVENPY"
    return pd.DataFrame({k: gsw[f"{pref}{k:02d}"] for k in scadenze}).astype(float)


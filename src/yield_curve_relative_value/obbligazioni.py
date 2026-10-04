"""Matematica di un titolo a cedola fissa del Tesoro USA.

Convenzioni (le stesse del mercato Treasury, dichiarate qui e verificate nei test):
- cedole semestrali, date generate all'indietro dalla scadenza ogni 6 mesi;
  se la scadenza e' a fine mese le date cedola sono a fine mese;
- rateo ACT/ACT: giorni trascorsi dall'ultima cedola su giorni del periodo cedolare;
- rendimento "Street": il primo periodo e' frazionario, w = giorni alla prossima
  cedola / giorni del periodo, poi si sconta con (1 + y/2) per periodo;
- prezzo per 100 di nominale; cedola e rendimento in percentuale annua (4.5 = 4,5%),
  composti semestralmente.

Il regolamento deve cadere prima della scadenza.
"""
import calendar
from datetime import date

from scipy.optimize import brentq


def _data_cedola(scadenza: date, mesi_prima: int) -> date:
    """Data cedola `mesi_prima` mesi prima della scadenza, mantenendo il giorno
    della scadenza (ridotto all'ultimo giorno del mese se il mese e' piu' corto)."""
    indice = scadenza.year * 12 + scadenza.month - 1 - mesi_prima
    anno, mese = divmod(indice, 12)
    mese += 1
    ultimo = calendar.monthrange(anno, mese)[1]
    if scadenza.day == calendar.monthrange(scadenza.year, scadenza.month)[1]:
        return date(anno, mese, ultimo)
    return date(anno, mese, min(scadenza.day, ultimo))


def date_cedole(scadenza: date, regolamento: date) -> tuple[date, list[date]]:
    """Restituisce (ultima cedola prima o al regolamento, cedole future fino alla scadenza)."""
    if regolamento >= scadenza:
        raise ValueError("il regolamento deve precedere la scadenza")
    k = 0
    while _data_cedola(scadenza, 6 * (k + 1)) > regolamento:
        k += 1
    precedente = _data_cedola(scadenza, 6 * (k + 1))
    future = [_data_cedola(scadenza, 6 * j) for j in range(k, -1, -1)]
    return precedente, future


def rateo(cedola: float, scadenza: date, regolamento: date) -> float:
    """Rateo di interesse maturato, per 100 di nominale (ACT/ACT)."""
    precedente, future = date_cedole(scadenza, regolamento)
    periodo = (future[0] - precedente).days
    return cedola / 2 * (regolamento - precedente).days / periodo


def flussi(cedola: float, scadenza: date, regolamento: date) -> list[tuple[date, float]]:
    """Flussi futuri (data, importo) per 100 di nominale; l'ultimo include il rimborso."""
    _, future = date_cedole(scadenza, regolamento)
    importi = [cedola / 2] * len(future)
    importi[-1] += 100.0
    return list(zip(future, importi, strict=True))


def _periodi(scadenza: date, regolamento: date) -> tuple[float, int]:
    """Quota w del primo periodo e numero di cedole future."""
    precedente, future = date_cedole(scadenza, regolamento)
    w = (future[0] - regolamento).days / (future[0] - precedente).days
    return w, len(future)


def prezzo_sporco(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Prezzo sporco (tel quel) dato il rendimento a scadenza."""
    w, n = _periodi(scadenza, regolamento)
    v = 1 + rendimento / 200
    totale = sum(cedola / 2 / v ** (w + k) for k in range(n))
    return totale + 100 / v ** (w + n - 1)


def prezzo_pulito(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Prezzo pulito = prezzo sporco meno rateo."""
    return prezzo_sporco(rendimento, cedola, scadenza, regolamento) - rateo(cedola, scadenza, regolamento)


def rendimento(prezzo: float, cedola: float, scadenza: date, regolamento: date, pulito: bool = True) -> float:
    """Rendimento a scadenza (% annuo, semestrale) dato il prezzo. Metodo di bisezione/Brent."""
    sporco = prezzo + rateo(cedola, scadenza, regolamento) if pulito else prezzo

    def errore(y: float) -> float:
        return prezzo_sporco(y, cedola, scadenza, regolamento) - sporco

    return brentq(errore, -5.0, 100.0, xtol=1e-12)


def duration_macaulay(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Duration di Macaulay in anni."""
    w, n = _periodi(scadenza, regolamento)
    v = 1 + rendimento / 200
    flussi_ = [cedola / 2] * n
    flussi_[-1] += 100
    valori = [f / v ** (w + k) for k, f in enumerate(flussi_)]
    tempi = [(w + k) / 2 for k in range(n)]
    return sum(t * pv for t, pv in zip(tempi, valori, strict=True)) / sum(valori)


def duration_modificata(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Duration modificata = Macaulay / (1 + y/2): variazione relativa del prezzo sporco per unita' di y."""
    return duration_macaulay(rendimento, cedola, scadenza, regolamento) / (1 + rendimento / 200)


def dv01(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Variazione del prezzo (per 100 di nominale) per +1 punto base di rendimento, in valore assoluto."""
    sporco = prezzo_sporco(rendimento, cedola, scadenza, regolamento)
    return duration_modificata(rendimento, cedola, scadenza, regolamento) * sporco * 1e-4


def convessita(rendimento: float, cedola: float, scadenza: date, regolamento: date) -> float:
    """Convessita' = (d2P/dy2) / P, con y in decimale (non in percentuale)."""
    w, n = _periodi(scadenza, regolamento)
    v = 1 + rendimento / 200
    flussi_ = [cedola / 2] * n
    flussi_[-1] += 100
    derivata = sum(f * (w + k) * (w + k + 1) / 4 / v ** (w + k + 2) for k, f in enumerate(flussi_))
    return derivata / prezzo_sporco(rendimento, cedola, scadenza, regolamento)

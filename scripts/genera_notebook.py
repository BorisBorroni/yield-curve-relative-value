"""Genera ed esegue notebooks/analisi.ipynb (gli output restano salvati nel file).

Uso: python scripts/genera_notebook.py   (richiede: pip install -e ".[notebook]")
Prerequisiti: data/storico/ e, per le tabelle del mondo sintetico, aver eseguito scripts/esegui_sintetico.py.
Il calcolo giornaliero della curva (2-3 minuti) viene riusato da output/ns_residui_bp.csv se esiste.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "analisi.ipynb"


def md(testo):
    return nbformat.v4.new_markdown_cell(testo.strip())


def codice(testo):
    return nbformat.v4.new_code_cell(testo.strip())


CELLE = [
    md("""
# Curva dei Treasury USA: Nelson-Siegel e valore relativo

Questo notebook ripercorre il progetto in ordine: matematica obbligazionaria, curva, mondo sintetico,
dati reali, strategia. Le tabelle sono prodotte dal codice del pacchetto; il testo spiega cosa guardare.
Tassi in percentuale, differenze in punti base (bp).
"""),
    codice("""
import sys
from pathlib import Path
from datetime import date

RADICE = Path.cwd().parent
sys.path.insert(0, str(RADICE / "src"))

import numpy as np
import pandas as pd
from IPython.display import display

from yield_curve_relative_value import analisi as an
from yield_curve_relative_value import dati, grafici
from yield_curve_relative_value import curva as cv
from yield_curve_relative_value import obbligazioni as ob
from yield_curve_relative_value import strategia as sg

pd.set_option("display.width", 150)
cmt, gsw = dati.carica_storico()
par_t = dati.par_cmt(cmt).dropna(how="all")
par_f = dati.par_gsw(gsw)
print(f"Tesoro: {len(par_t)} giorni dal {par_t.index[0].date()} al {par_t.index[-1].date()}")
"""),
    md("""
## 1. Matematica di un titolo

Un titolo a cedola semestrale: prezzo sporco dato il rendimento, rateo ACT/ACT, prezzo pulito, e le misure
di sensibilita'. Qui un esempio con date fisse; il rendimento e' ricalcolato dal prezzo come controllo.
"""),
    codice("""
reg, scad, cedola, y = date(2025, 6, 30), date(2035, 5, 15), 4.25, 4.40
sporco = ob.prezzo_sporco(y, cedola, scad, reg)
esempio = pd.Series({
    "prezzo sporco": sporco,
    "rateo": ob.rateo(cedola, scad, reg),
    "prezzo pulito": ob.prezzo_pulito(y, cedola, scad, reg),
    "rendimento ricalcolato": ob.rendimento(ob.prezzo_pulito(y, cedola, scad, reg), cedola, scad, reg),
    "duration modificata": ob.duration_modificata(y, cedola, scad, reg),
    "DV01 (per 100)": ob.dv01(y, cedola, scad, reg),
    "convessita'": ob.convessita(y, cedola, scad, reg),
})
display(esempio.round(4).to_frame("valore"))
"""),
    md("""
## 2. La curva e il controllo contro la Fed

La formula di Svensson con i parametri pubblicati dalla Fed deve riprodurre zero, forward e par yield che la
Fed pubblica. Qui un campione di 40 date dal 1971 (stesso file usato dai test). Lo scarto massimo e' in punti
percentuali; la Fed arrotonda a 4 decimali.
"""),
    codice("""
camp = pd.read_csv(RADICE / "tests" / "dati" / "gsw_campione.csv", index_col=0, parse_dates=True)
n = np.arange(1, 31)
scarti = {"zero": 0.0, "forward": 0.0, "par yield": 0.0}
for _, r in camp.iterrows():
    p = cv.Parametri(r.BETA0, r.BETA1, r.BETA2, r.BETA3, r.TAU1, r.TAU2 if r.BETA3 != 0 else 5.0)
    for nome, pref, calc in (("zero", "SVENY", cv.rendimento_zero(n, p)), ("forward", "SVENF", cv.forward(n, p)),
                             ("par yield", "SVENPY", np.array([cv.par_yield(k, p) for k in n]))):
        pub = np.array([r[f"{pref}{k:02d}"] for k in n])
        scarti[nome] = max(scarti[nome], float(np.nanmax(np.abs(calc - pub))))
display(pd.Series(scarti).to_frame("scarto massimo (punti %)"))
"""),
    md("""
## 3. Mondo sintetico

Prima dei dati veri: una curva Nelson-Siegel nota, con mispricing e rumore noti. Serve a sapere quanto deve
essere grande un mispricing per essere distinguibile dal rumore, e a vedere i falsi positivi. Le tabelle
vengono da `scripts/esegui_sintetico.py`.
"""),
    codice("""
for nome in ("recupero", "falsi_positivi", "potere"):
    f = RADICE / "output" / f"sintetico_{nome}.csv"
    if f.exists():
        print(nome)
        display(pd.read_csv(f, index_col=0))
    else:
        print(f"{f.name} non trovato: esegui scripts/esegui_sintetico.py")
"""),
    md("""
## 4. Premio del titolo nuovo: Tesoro meno Fed

I punti del Tesoro sono costruiti con i soli titoli appena emessi; la curva della Fed li esclude. La differenza
misura quanto costano di piu' i titoli nuovi (segno negativo = rendimento piu' basso). Il Tesoro ha cambiato
metodo di costruzione il 6 dicembre 2021, quindi i periodi si leggono separati. Il t e' di Newey-West.
"""),
    codice("""
premio = an.differenza_bp(par_t, par_f)
display(an.riepilogo(premio).drop(index="intero", level=0))
display(grafici.grafico_premio(premio))
"""),
    md("""
## 5. Nelson-Siegel sui punti del Tesoro

La curva si adatta ogni giorno ai par yield (non ai tassi zero) e il residuo e' il valore osservato meno la
curva. Se i residui fossero rumore non dovrebbero avere memoria: sotto si vede la loro autocorrelazione a un
giorno.
"""),
    codice("""
f = RADICE / "output" / "ns_residui_bp.csv"
if f.exists():
    residui = pd.read_csv(f, index_col=0, parse_dates=True)
    residui.columns = [int(c) for c in residui.columns]
else:
    residui = an.adatta_serie(par_t)[1]
rmse = pd.DataFrame({k: np.sqrt((v ** 2).mean()) for k, v in an.periodi(residui).items()}).T.round(2)
display(rmse)
display(pd.DataFrame({"autocorrelazione a 1 giorno": {c: residui[c].autocorr(1) for c in residui.columns}}).T.round(2))
"""),
    md("""
## 6. Componenti principali contro fattori, e previsione

Le tre componenti principali delle variazioni giornaliere contro i tre fattori di Nelson-Siegel (livello,
pendenza, curvatura con lambda di Diebold-Li). Poi la previsione fuori campione dei fattori: AR(1) contro
passeggiata casuale.
"""),
    codice("""
completi = par_t.dropna()
var, punteggi, _ = an.pca(completi.diff().dropna() * 100)
fattori = an.fattori_dl(completi).diff().dropna()
display(pd.DataFrame({"varianza spiegata": var.round(3), "R2 sui 3 fattori": an.r2_su_fattori(punteggi, fattori).round(3).to_numpy()},
                     index=["PC1", "PC2", "PC3"]))
righe = []
for h in (1, 6, 12):
    rm, dm, k = an.previsione_dl(completi, h, "2010-01-01")
    righe.append({"orizzonte (mesi)": h, "previsioni": k, "RMSE AR(1)": rm["AR(1)"].mean().round(1),
                  "RMSE passeggiata casuale": rm["RW"].mean().round(1), "t Diebold-Mariano": dm})
display(pd.DataFrame(righe).set_index("orizzonte (mesi)"))
"""),
    md("""
## 7. Farfalle

La regola e' fissata prima: z-score su 60 giorni precedenti, soglia 1, posizione sul rientro. Farfalla 2-5-10
con pesi DV01 (0.5, -1, 0.5) e neutrale ai fattori, con tre livelli di costo per gamba. Il guadagno e' in bp
per unita' di DV01 al giorno.
"""),
    codice("""
righe = []
cum = None
for neutr in ("dv01", "fattori"):
    w = sg.pesi_farfalla((2, 5, 10), neutr)
    x = sg.serie_farfalla(par_t, (2, 5, 10), w)
    for costo in (0.0, 0.25, 0.5):
        g, _ = sg.guadagni_serie(x.to_numpy(), x.to_numpy(), abs(w).sum(), 60, 1.0, costo)
        righe.append({"neutralita": neutr, "costo (bp)": costo, "bp/giorno": g.mean().round(4), "bp/anno": (g.mean() * 252).round(1)})
    if neutr == "dv01":
        cum = grafici.guadagni_cumulati(x, abs(w).sum())
display(pd.DataFrame(righe).set_index(["neutralita", "costo (bp)"]))
display(grafici.grafico_guadagni(cum, "Farfalla 2-5-10: guadagno cumulato della regola z-score"))
"""),
    md("""
## 8. La curva di oggi

Il confronto fra l'ultimo giorno disponibile e lo storico non e' nel notebook, perche' dipende dal giorno in cui
si esegue: lo produce `python scripts/analisi_corrente.py`, che scarica i dati recenti in `data/corrente/`
(cartella non versionata) e stampa residui, z-score e farfalle.
"""),
    md("""
## Conclusioni

- Il residuo di Nelson-Siegel sui punti del Tesoro e' molto persistente: e' soprattutto errore di forma del
  modello e rumore di costruzione dei punti, non mispricing che rientra.
- La regola z-score guadagna al lordo, ma con un costo di circa 0.2 bp per unita' di DV01 il guadagno si azzera.
- Il premio del titolo nuovo e' invece una caratteristica stabile e misurabile della curva.

I limiti (punti interpolati, nessun prezzo eseguibile, costi ipotizzati) sono nel README.
"""),
]


def main():
    nb = nbformat.v4.new_notebook(cells=CELLE)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    DEST.parent.mkdir(exist_ok=True)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(DEST.parent)}})
    client.execute()
    nbformat.write(nb, DEST)
    print("Notebook salvato in", DEST)


if __name__ == "__main__":
    main()

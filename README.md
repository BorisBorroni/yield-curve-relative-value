# yield-curve-relative-value

Il modello di Nelson-Siegel sulla curva dei Treasury USA: controllo delle formule contro la curva della Fed e verifica di una strategia di valore relativo (statistico) sulla curva. Non la chiamo arbitraggio: non c'è nessun guadagno privo di rischio, solo una scommessa sul ritorno alla media di scostamenti dalla curva.

La domanda a cui voglio rispondere è: quando un punto della curva sta sopra o sotto il modello, che cosa significa quello scostamento (il residuo)? Può essere errore del modello, un premio noto del mercato, o un segnale che rientra e su cui si può guadagnare. E dopo i costi resta qualcosa? Ho scritto questo progetto per dare una risposta quantitativa, qualunque sia il segno.

## Indice

1. [Risultato in breve](#1-risultato-in-breve)
2. [Struttura del progetto](#2-struttura-del-progetto)
3. [Convenzioni](#3-convenzioni)
4. [Dati e fonti](#4-dati-e-fonti)
5. [Matematica di un titolo](#5-matematica-di-un-titolo)
6. [La curva e il controllo contro la Fed](#6-la-curva-e-il-controllo-contro-la-fed)
7. [Mondo sintetico](#7-mondo-sintetico)
8. [Dati reali](#8-dati-reali)
9. [Strategia](#9-strategia)
10. [La curva di oggi](#10-la-curva-di-oggi)
11. [Limiti](#11-limiti)
12. [Come riprodurre](#12-come-riprodurre)
13. [Bibliografia](#13-bibliografia)

## 1. Risultato in breve

- Le formule di Nelson-Siegel/Svensson, con i parametri pubblicati dalla Fed, riproducono zero, forward e par yield della Fed con uno scarto massimo di 0.00005 punti percentuali, cioè l'arrotondamento dei dati. È un controllo dell'implementazione delle formule, non una calibrazione sui dati della Fed.
- I punti del Tesoro hanno rendimenti più bassi della curva Fed: il segno è coerente con titoli appena emessi più cari, ma il confronto mescola altre differenze (sezione 8). A 10 anni la differenza media è -10.9 bp fino a dicembre 2021 e -5.7 dopo. Con errori standard prudenti (250 ritardi di Newey-West) |t| supera 2 in 12 celle su 16 (scadenza e periodo). Le eccezioni sono 2 e 30 anni fino a dicembre 2021 (t -1.55 e -1.99), 1 e 7 anni da dicembre 2021 (t -1.51 e 0.96).
- Il residuo di Nelson-Siegel sui punti del Tesoro è di 2.5-3.6 bp fino a 7 anni e 6-9 bp a 10-30 anni, ed è molto persistente (autocorrelazione a un giorno 0.91-0.995). Svensson ne toglie una parte sistematica solo a 20 e 30 anni, non fra 1 e 10 anni. Non ho separato forma della curva, differenze di metodo del Tesoro e premio di mercato.
- La regola di ritorno alla media sui residui guadagna 0.099 bp al giorno al lordo se si scambia allo stesso prezzo che genera il segnale. Un residuo stazionario e persistente come quello reale, senza alcun mispricing, ne guadagnerebbe 0.052.
- Se si scambia il giorno dopo il segnale, il guadagno lordo reale scende a 0.038, meno dei 0.049 attesi senza mispricing, e con un costo di 0.25 bp è negativo. Le farfalle 2-5-10 e 5-10-30 pareggiano con costi di 0.04-0.2 bp per unità di DV01 su ogni gamba. Non ho trovato un vantaggio eseguibile.

Premio del titolo nuovo: media mobile di un anno della differenza fra punti del Tesoro e curva Fed, a 10, 20 e 30 anni.

![Premio del titolo nuovo](img/premio_on_the_run.png)

Guadagno cumulato della regola sulla farfalla 2-5-10 con pesi DV01: lordo con scambio nello stesso giorno, lordo con scambio il giorno dopo, e con il giorno dopo e un costo di 0.25 bp.

![Guadagno cumulato della farfalla](img/guadagni_farfalla.png)

## 2. Struttura del progetto

```
src/yield_curve_relative_value/
    obbligazioni.py     prezzo, rateo, rendimento, duration, DV01, convessità
    curva.py            Nelson-Siegel / Svensson, par yield, calibrazione
    mondo_sintetico.py  curva nota con mispricing e rumore noti
    dati.py             lettura, download (con robots.txt) e storico
    analisi.py          premio del titolo nuovo, fit giornaliero, PCA, previsione
    strategia.py        z-score, farfalle, guadagni
    corrente.py         confronto della curva più recente con lo storico
    statistica.py       Newey-West
    grafici.py          i due grafici
scripts/                un script per ogni sezione dei risultati
tests/                  85 test; quelli con simulazioni sono marcati "slow"
data/storico/           curve del Tesoro e della Fed dal 2000 al 2025 (CSV piccoli, versionati)
data/corrente/          dati recenti scaricati a ogni esecuzione (non versionata)
notebooks/analisi.ipynb notebook eseguito, con gli output salvati
img/                    i grafici di questo README
```

Ogni numero di questo README viene da uno script o dal notebook (tabella a fine sezione 12); dove non è una tabella salvata in `output/`, lo script lo stampa. Le formule sono scritte nelle docstring.

## 3. Convenzioni

Le convenzioni sono dichiarate nel codice e controllate dai test dove si può, perché sbagliarle dà errori che sembrano risultati. La convenzione "Street" con primo periodo frazionario è controllata solo con l'andata e ritorno prezzo-rendimento, non con un valore indipendente.

| Cosa | Convenzione |
|---|---|
| Rendimenti | percentuale annua (4.5 = 4,5%) |
| Differenze | punti base (bp) |
| Cedole dei titoli | semestrali; date generate all'indietro dalla scadenza; scadenza a fine mese implica cedole a fine mese |
| Rateo | ACT/ACT: giorni trascorsi sul periodo cedolare |
| Rendimento di un titolo | stile "Street": primo periodo frazionario, poi sconto con (1 + y/2) per periodo |
| Tassi zero (curva Fed, mia curva) | capitalizzazione continua |
| Par yield | rendimento della cedola semestrale che fa valere il titolo 100: c = 200 (1 - P(T)) / somma P(t_i), t_i = 0.5, 1, ..., T |
| Tempo dei flussi nella curva | giorni reali / 365.25 |
| Punti del Tesoro | par yield: la curva si adatta in modo che i suoi par yield riproducano quelli osservati |

Lo zero continuo e il par yield sono due cose diverse e la Fed li pubblica in colonne separate (SVENY e SVENPY): confonderli è una trappola frequente. Il prezzo di un titolo a cedola uguale al par yield viene 100.004 e non 100, perché il par yield usa mezzi anni esatti e il prezzo i giorni reali; il test lo accetta con tolleranza 0.01 e spiega lo scarto in un commento.

## 4. Dati e fonti

Uso solo dati gratuiti.

| Fonte | Cosa | Uso |
|---|---|---|
| Tesoro USA, punti a scadenza costante | par yield giornalieri a 1, 2, 3, 5, 7, 10, 20, 30 anni, dal 2000 | dato principale |
| Fed, curva di Gurkaynak-Sack-Wright | parametri e curve di Svensson stimati su singoli titoli, dal 1961 | riferimento esterno |

Perché queste: il Tesoro pubblica solo punti interpolati (nessun prezzo di singolo titolo), e la Fed è l'unico riferimento stimato su singoli titoli che sia gratuito e documentato. Non ho trovato prezzi giornalieri gratuiti e utilizzabili per singolo titolo (FedInvest li pubblica ma sono valutazioni amministrative, vedi sotto), quindi non faccio un backtest a livello di obbligazione.

Cose da sapere:

- Il Tesoro costruisce i punti con i soli titoli appena emessi (bills, note 2-3-5-7-10 anni, bond 20 e 30 anni), con quotazioni indicative delle 15:30. Dal 6 dicembre 2021 usa il metodo monotone convex, prima uno spline quasi-cubico: è una rottura strutturale, quindi leggo sempre due periodi.
- La curva Fed esclude bills, titoli sotto 3 mesi, callable, il titolo nuovo e il primo vecchio. Gli autori avvertono che i parametri saltano da un giorno all'altro: confronto curve, mai parametri.
- I 30 anni mancano dal 2002 al 2006 (non erano emessi); in quei giorni il fit usa le altre sette scadenze.
- Prima di ogni download leggo il robots.txt dell'host: 404 vuol dire nessuna regola, 401 o 403 vietato. Una richiesta al secondo.
- Ho scartato FedInvest (TreasuryDirect): i prezzi sono valutazioni amministrative per titoli speciali, tutte multiple di 1/32 (risoluzione 0.4-1.6 bp), e il sito vieta robot e ridistribuzione. I termini d'uso delle altre fonti li ho letti ma non sono un legale: chi riusa i dati controlli le condizioni.

## 5. Matematica di un titolo

`obbligazioni.py`: prezzo sporco dato il rendimento, rateo, prezzo pulito, rendimento dal prezzo (Brent), duration di Macaulay e modificata, DV01 (variazione del prezzo per 100 per +1 bp) e convessità. Controllo con formule chiuse (titolo alla pari su data cedola, rendita, zero coupon) e con differenze finite per DV01 e convessità.

## 6. La curva e il controllo contro la Fed

`curva.py`: zero, fattore di sconto, forward istantaneo, par yield e carichi dei tre fattori. Formula di Svensson (Nelson-Siegel è il caso beta3 = 0):

y(n) = b0 + b1 A(n/t1) + b2 [A(n/t1) - exp(-n/t1)] + b3 [A(n/t2) - exp(-n/t2)], con A(x) = (1 - exp(-x)) / x.

Calibrazione: sui par yield osservati (quella che uso per i risultati sui dati reali), con minimi quadrati non pesati e più punti di partenza perché la stima non lineare ha minimi locali (ripartire dalla soluzione del giorno prima peggiorava i residui). Nel pacchetto c'è anche la calibrazione sui prezzi, con pesi 1/duration come Gurkaynak, Sack e Wright, e tau libero o fisso (Diebold-Li, tau fisso a 1.37 anni, cioè lambda 0.0609 al mese): è codice di libreria, controllato con titoli sintetici, e non entra in nessun risultato reale.

Controllo: con i parametri della Fed su 40 date dal 1971 al 2026 (file `tests/dati/gsw_campione.csv`) ricalcolo zero, forward e par yield e li confronto con le colonne pubblicate. Scarto massimo 0.00005 punti percentuali per tutte e tre (lo stampa il notebook; il test usa la tolleranza 1e-4). Diagnostica di identificabilità: `condizionamento` misura la collinearità dei carichi.

## 7. Mondo sintetico

Prima dei dati veri: una curva Nelson-Siegel nota con fattori AR(1), a cui sommo un mispricing OU (mean-reverting, dev. std ed emivita note) e un rumore di misura. Qui so qual è la risposta giusta.

Le impostazioni sono fisse: 1500 giorni, emivita 5 giorni, rumore 0.5 bp, 60 simulazioni per cella (100 nella tabella del potere), soglia t di Newey-West 2 (10 ritardi). Il rumore è errore di misura: disturba il segnale ma i guadagni si calcolano sui prezzi senza rumore, altrimenti ogni rumore indipendente darebbe un guadagno fasullo perché rientra da solo.

Recupero con rumore (media di 60 simulazioni, errore quadratico medio):

| tau | beta0 | beta1 | beta2 | tau | curva (bp) |
|---|---|---|---|---|---|
| libero | 0.012 | 0.027 | 0.411 | 0.648 | 0.363 |
| fisso = vero (2.0) | 0.005 | 0.009 | 0.027 | 0.000 | 0.306 |
| fisso Diebold-Li (1.37) | 0.057 | 0.302 | 0.910 | 0.630 | 2.331 |

I parametri si stimano male (beta2 e tau soprattutto) ma la curva è precisa. Anche gli autori della curva Fed avvertono che i parametri sono instabili (sezione 4). Un tau fisso sbagliato costa 2.3 bp di curva.

Falsi positivi (nessun mispricing, guadagni sui prezzi senza rumore):

| curva vera | fit | costo (bp) | frequenza t > 2 | guadagno medio (bp/giorno) |
|---|---|---|---|---|
| Nelson-Siegel | tau libero | 0 | 0.983 | 0.0114 |
| Nelson-Siegel | tau libero | 0.25 | 0.000 | -0.1185 |
| Nelson-Siegel | tau fisso | 0 | 0.017 | -0.0003 |
| Nelson-Siegel | tau fisso | 0.25 | 0.000 | -0.1311 |
| Svensson | tau libero | 0 | 1.000 | 0.0778 |
| Svensson | tau libero | 0.25 | 0.000 | -0.0212 |
| Svensson | tau fisso | 0 | 0.117 | 0.0027 |
| Svensson | tau fisso | 0.25 | 0.000 | -0.0903 |

Stimare il tau ogni giorno sugli stessi dati crea un guadagno lordo minuscolo ma "significativo" quasi sempre; con tau fisso e curva vera Nelson-Siegel i falsi positivi tornano al livello atteso (0.017), con una seconda gobba restano 0.117. Se la curva vera ha una seconda gobba che Nelson-Siegel non descrive, il guadagno lordo apparente sale. In tutti i casi sparisce con un costo di 0.25 bp.

I punti del Tesoro hanno due decimali, cioè 1 bp di risoluzione. Ho ripetuto falsi positivi e potere arrotondando anche i dati sintetici a 1 bp (l'errore di arrotondamento conta come rumore non negoziabile). A costo 0.25 i falsi positivi restano 0, ma il potere a 1.75 bp di mispricing scende da 0.90 a 0.46 con tau libero e da 0.72 a 0.15 con tau fisso: l'arrotondamento rende più difficile vedere un mispricing piccolo. I numeri che seguono sono quindi calcolati con l'arrotondamento a 1 bp, come i dati reali.

Mispricing minimo rilevabile, cioè la dev. std più piccola della griglia (0.1, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 3, 3.25, 3.5, 4, 5 bp) per cui il potere è almeno 0.8, 100 simulazioni per cella. Fra parentesi il valore senza arrotondamento:

| fit | esecuzione | costo 0 | costo 0.25 bp | costo 0.5 bp |
|---|---|---|---|---|
| tau libero | stesso giorno | non interpretabile | 2.0 (1.75) | 3.25 (3.25) |
| tau fisso 2.0 | stesso giorno | 0.5 (0.25) | 2.0 (2.0) | 3.25 (3.0) |
| tau fisso 2.0 | giorno dopo | 0.5 (0.25) | 2.5 (2.5) | 4.0 (4.0) |

Con tau libero e costo 0 il test "rileva" già a 0.1 bp, ma è il bias dei falsi positivi della tabella sopra (0.98 senza alcun mispricing): non è potere. Le soglie valgono al passo della griglia e alcune sono vicine al limite (per esempio 3.0 con tau fisso e costo 0.5 vale 0.82 senza arrotondamento e 0.51 con); con semi diversi possono spostarsi di un passo. In ordine di grandezza: con costi di 0.25-0.5 bp un mispricing deve avere una dev. std di circa 2-4 bp per essere distinguibile dal rumore, e se i residui reali fossero più piccoli nessuna strategia sarebbe distinguibile.

## 8. Dati reali

Premio del titolo nuovo: Tesoro meno curva Fed, per scadenza e periodo (media e deviazione standard in bp; t di Newey-West con 250 ritardi). Sul periodo intero le serie hanno autocorrelazione a 60 giorni fra 0.49 e 0.93 (dopo dicembre 2021 è più bassa, fra -0.22 e 0.58), quindi pochi ritardi sovrastimano il t: con 20 ritardi a 10 anni fino a dicembre 2021 avrei letto -16.1 invece di -5.0. Su 1017 giorni, il sottoperiodo più corto, 250 ritardi sono molti e quindi prudenti (tendono a sottostimare il t). I p-value non sono corretti per test multipli.

| scadenza | fino a dic 2021: media | dev. std | t | da dic 2021: media | dev. std | t |
|---|---|---|---|---|---|---|
| 1 | -3.62 | 7.25 | -3.16 | -1.71 | 4.98 | -1.51 |
| 2 | -0.83 | 3.62 | -1.55 | -3.53 | 2.74 | -4.91 |
| 3 | -1.82 | 4.11 | -2.92 | -2.89 | 2.28 | -4.47 |
| 5 | -1.95 | 3.32 | -3.85 | -2.46 | 1.61 | -5.37 |
| 7 | -2.60 | 6.30 | -2.37 | 0.22 | 1.80 | 0.96 |
| 10 | -10.86 | 11.00 | -5.02 | -5.66 | 3.01 | -6.42 |
| 20 | -5.87 | 5.32 | -5.68 | 4.99 | 3.05 | 4.95 |
| 30 | -3.10 | 7.79 | -1.99 | -11.59 | 6.29 | -5.33 |

Il segno medio è negativo (titolo nuovo più caro), con il valore più grande a 10 anni fino a dicembre 2021 (-10.9) e a 30 anni dopo (-11.6). A 20 anni cambia segno dopo il 2021 e a 30 anni il premio si allarga: non ho una spiegazione verificata. Possono contribuire il metodo del Tesoro, l'orario (15:30 contro fine giornata) e la selezione dei titoli nella curva Fed. Il confronto di fondo è tra due stime diverse della stessa curva, quindi una parte del premio è differenza di metodo e non premio di mercato.

Nelson-Siegel sui punti del Tesoro (par yield, adattamento giornaliero, tau tra 0.3 e 10 anni). Scarto quadratico medio del residuo in bp:

| periodo | 1 | 2 | 3 | 5 | 7 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| fino a dic 2021 | 2.54 | 3.19 | 2.48 | 3.04 | 3.79 | 5.61 | 8.12 | 6.31 |
| da dic 2021 | 1.99 | 2.56 | 3.23 | 1.71 | 2.33 | 7.17 | 12.04 | 6.96 |
| intero | 2.46 | 3.10 | 2.61 | 2.87 | 3.60 | 5.88 | 8.85 | 6.43 |

Il tau tocca il limite superiore (10 anni) in 433 giorni su 6502 e quello inferiore in 31: i parametri sono instabili. I residui hanno autocorrelazione a un giorno di 0.911, 0.933, 0.923, 0.958, 0.974, 0.984, 0.995, 0.990 (nelle scadenze in ordine): non sono rumore.

Curva Nelson-Siegel (fatta sul Tesoro) contro curva Fed, scarto quadratico medio in bp: 7.31 a 1 anno, 4.53 a 2, 4.46 a 3, 4.00 a 5, 6.48 a 7, 11.34 a 10, 10.01 a 20, 10.31 a 30 (intero periodo; la tabella per periodo è in `output/ns_contro_fed.csv`). Dopo dicembre 2021 lo scarto scende a 1, 5, 7, 10, 20 e 30 anni (a 10 anni da 12.27 a 3.18 bp) ma sale a 2 e 3 anni (da 4.47 a 4.86 e da 4.26 a 5.38).

Componenti principali delle variazioni giornaliere (5506 variazioni con tutte le scadenze, calcolate in modo da non attraversare il buco dei 30 anni del 2002-2006): 84.3%, 11.2%, 2.4% della varianza. I tre fattori di Nelson-Siegel (livello, pendenza, curvatura con il tau di Diebold-Li) spiegano le tre componenti con R2 di 1.000, 1.000 e 0.996.

Previsione fuori campione dei fattori (finestra espansiva, test dal 2010, calendario mensile senza coppie che attraversano il buco del 2002-2006), AR(1) contro passeggiata casuale; errore quadratico medio sulle scadenze, in bp:

| orizzonte (mesi) | previsioni | AR(1) | passeggiata casuale | t Diebold-Mariano (positivo = AR(1) meglio) | t Clark-West |
|---|---|---|---|---|---|
| 1 | 191 | 24.2 | 23.3 | -2.32 | -0.68 |
| 6 | 186 | 70.2 | 62.9 | -1.91 | -0.29 |
| 12 | 180 | 111.4 | 96.8 | -1.60 | -0.05 |

L'AR(1) non batte la passeggiata casuale, e il test di Clark-West, quello adatto a modelli annidati (la passeggiata casuale è un caso particolare dell'AR(1)), non rifiuta mai la passeggiata casuale (si rifiuterebbe sopra 1.645). Una spiegazione possibile è che l'AR(1) tira i fattori verso la media dei primi anni e sbaglia nel regime successivo, ma non l'ho testata. Il periodo contiene il tasso zero e lo shock del 2022 e le previsioni a 6 e 12 mesi si sovrappongono, quindi il test ha poca potenza. Lo riporto così com'è.

## 9. Strategia

Le ipotesi e i parametri sono scritti in testa a `scripts/esegui_strategia.py`. Sono quattro e i p-value non sono corretti per test multipli. I parametri (finestra 60 giorni, soglia 1, costi 0 / 0.25 / 0.5 bp, tau di Diebold-Li 1.37 anni, 10 ritardi di Newey-West) sono convenzionali e non sono stimati sui dati, ma non posso provare di averli fissati prima di vedere i dati. Dopo una prima lettura dei risultati ho aggiunto tre controlli senza cambiare i criteri di H1-H4: l'esecuzione il giorno dopo il segnale, un controllo nullo calibrato sui residui reali, e il t con 60 ritardi accanto a quello con 10.

Regola: z-score = (valore - media) / deviazione standard sui 60 giorni precedenti (il giorno corrente escluso). Posizione +1 sopra la soglia, -1 sotto, 0 altrimenti. Il guadagno del giorno t+1 è posizione per (valore di oggi - valore di domani) in bp, meno il costo. Il costo è in bp di rendimento per ogni unità di DV01 scambiata su ogni gamba. Questa è l'esecuzione "stesso giorno": si compra al valore stesso che ha generato il segnale, una lettura interpolata delle 15:30 il cui errore di misura rientra da solo. L'esecuzione "giorno dopo" apre la posizione un giorno più tardi (guadagno = posizione per valore del giorno t+1 meno valore del giorno t+2): è una prova grezza, non una simulazione di esecuzione reale.

Farfalle: 2-5-10 e 5-10-30. Con pesi DV01 le ali pesano 0.5 e 0.5 e la pancia -1. Con pesi "neutrali ai fattori" le ali sono scelte per annullare l'esposizione al livello e alla pendenza di Nelson-Siegel (pesi 0.33 e 0.67 per la 2-5-10, 0.41 e 0.59 per la 5-10-30): resta solo la curvatura. Il 5-10-30 parte dal 2006 per il buco dei 30 anni.

**H1, errore di modello: confermata solo in parte.** Rapporto tra lo scarto di Svensson (6 parametri) e quello di Nelson-Siegel (4 parametri), 8 punti: 0.53 sui dati reali (0.52 fino a dic 2021, 0.57 dopo); con curva vera Nelson-Siegel e solo rumore di 3 bp, 0.75. Per scadenza, dati reali contro solo rumore (media su 10 simulazioni di 120 giorni):

| scadenza (anni) | 1 | 2 | 3 | 5 | 7 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| dati reali | 0.59 | 0.75 | 0.90 | 0.78 | 0.99 | 0.67 | 0.27 | 0.19 |
| solo rumore | 0.48 | 0.65 | 0.84 | 0.80 | 0.97 | 0.71 | 0.55 | 0.48 |

Fra 1 e 10 anni i dati reali stanno al livello del solo rumore o sopra (Svensson non toglie nulla di sistematico). A 20 e 30 anni stanno molto sotto (0.27 e 0.19 contro 0.55 e 0.48): lì Svensson toglie più di quanto farebbe il solo sovradattamento di 6 parametri su 8 punti. Che cosa tolga non lo so: può essere forma reale della curva a lungo termine, o una caratteristica di quelle scadenze (interpolazione del Tesoro, liquidità). Dopo dicembre 2021 su alcuni punti Svensson sbaglia di più di Nelson-Siegel (1.16 a 1 anno, 1.55 a 7): la somma dei quadrati su tutti i punti di un giorno non supera mai quella di Nelson-Siegel (0 giorni su 6502), quindi non è un guasto dell'ottimizzatore ma una redistribuzione dell'errore fra i punti.

**H2, residui di Nelson-Siegel (media delle 8 scadenze, costo su una sola gamba): criterio soddisfatto con lo scambio nello stesso giorno, non accettato come prova.** Bp al giorno (intero periodo), con t di Newey-West a 10 e a 60 ritardi:

| esecuzione | costo (bp) | bp/giorno | t (10 ritardi) | t (60 ritardi) |
|---|---|---|---|---|
| stesso giorno | 0 | 0.0985 | 20.90 | 16.20 |
| stesso giorno | 0.25 | 0.0362 | 8.40 | 6.28 |
| stesso giorno | 0.5 | -0.0261 | -6.35 | -4.50 |
| giorno dopo | 0 | 0.0377 | 9.86 | 8.46 |
| giorno dopo | 0.25 | -0.0245 | -6.31 | -5.10 |
| giorno dopo | 0.5 | -0.0868 | -20.67 | -15.65 |

Il criterio dichiarato (netto a 0.25 bp positivo con t > 2) è soddisfatto con lo scambio nello stesso giorno, e questo vale in entrambi i periodi (+0.0363 fino a dic 2021, +0.0358 dopo). Non lo leggo come vantaggio, per tre ragioni. Primo, il costo è su una gamba sola, senza coprire la curva. Secondo, con lo scambio il giorno dopo il guadagno lordo scende da 0.0985 a 0.0377 e a 0.25 bp diventa negativo. Terzo, un guadagno lordo positivo è atteso da qualunque residuo stazionario e persistente, perché la regola scommette sul ritorno alla media, e va confrontato con quello e non con zero. Il controllo: residui simulati senza alcun mispricing, AR(1) con la stessa persistenza (0.91-0.995 per scadenza) e la stessa deviazione standard (2.4-8.1 bp) di quelli reali, stessa lunghezza, stessa regola, 200 simulazioni. Bp al giorno su tutte le osservazioni:

| esecuzione | costo (bp) | reale | nullo: media | nullo: 5-95% | quota di nulli >= reale |
|---|---|---|---|---|---|
| stesso giorno | 0 | 0.0988 | 0.0521 | 0.0489-0.0557 | 0.0 |
| stesso giorno | 0.25 | 0.0364 | 0.0085 | 0.0057-0.0111 | 0.0 |
| giorno dopo | 0 | 0.0378 | 0.0488 | 0.0452-0.0526 | 1.0 |
| giorno dopo | 0.25 | -0.0247 | 0.0052 | 0.0017-0.0087 | 1.0 |

Con lo scambio nello stesso giorno il guadagno reale supera tutti i nulli. L'eccedenza (circa 0.05 bp) è coerente con una parte del residuo che rientra subito e non è negoziabile (errore di interpolazione o di misura), ma anche con la stima giornaliera del tau: in un mondo sintetico con una seconda gobba e senza mispricing il guadagno lordo apparente con tau libero è 0.078 (sezione 7). Non ho separato queste cause. Con lo scambio il giorno dopo il guadagno reale sta sotto tutti i nulli: i residui reali non guadagnano più di un residuo stazionario senza mispricing. I valori "reale" qui sono medie su tutte le osservazioni; nella tabella H2 sono medie di medie giornaliere, per questo differiscono di qualche decimillesimo. Il pareggio dei costi è circa 0.40 bp (stesso giorno) e 0.15 bp (giorno dopo). Il nullo è un AR(1) per scadenza, quindi non cattura la dipendenza fra scadenze né le code. Non riporto lo Sharpe: la media di 8 serie legate fra loro (i residui di un adattamento a 4 parametri su 8 punti hanno vincoli lineari) e una sola gamba di costo lo renderebbero fuorviante.

**H3, farfalle: rifiutata.** Guadagno in bp al giorno (intero periodo), con il costo di pareggio, cioè il costo per cui il guadagno è zero:

| farfalla | neutralita | esecuzione | costo 0 | costo 0.25 | costo 0.5 | t, costo 0 (10 rit.) | t, costo 0.25 (10 rit.) | costo di pareggio |
|---|---|---|---|---|---|---|---|---|
| 2-5-10 | DV01 | stesso giorno | 0.0718 | -0.0201 | -0.1120 | 5.42 | -1.66 | 0.195 |
| 2-5-10 | fattori | stesso giorno | 0.0641 | -0.0237 | -0.1115 | 4.91 | -1.98 | 0.183 |
| 5-10-30 | DV01 | stesso giorno | 0.0583 | -0.0356 | -0.1294 | 5.08 | -3.48 | 0.155 |
| 5-10-30 | fattori | stesso giorno | 0.0467 | -0.0459 | -0.1384 | 4.06 | -4.39 | 0.126 |
| 2-5-10 | DV01 | giorno dopo | 0.0311 | -0.0608 | -0.1528 | 2.47 | -4.85 | 0.085 |
| 2-5-10 | fattori | giorno dopo | 0.0368 | -0.0510 | -0.1388 | 2.79 | -3.90 | 0.105 |
| 5-10-30 | DV01 | giorno dopo | 0.0184 | -0.0754 | -0.1691 | 1.64 | -6.69 | 0.049 |
| 5-10-30 | fattori | giorno dopo | 0.0161 | -0.0765 | -0.1690 | 1.42 | -6.69 | 0.043 |

Si resta in posizione circa il 51% dei giorni e la posizione cambia 0.18 volte al giorno. La versione neutrale ai fattori non batte in modo coerente quella DV01: è meglio in un solo caso su quattro (2-5-10, scambio il giorno dopo). Nell'analisi di sensibilità (finestre 20, 60, 120; soglie 0.5, 1, 1.5, 2; scambio nello stesso giorno, che è il caso più favorevole) tutte le 48 combinazioni sono negative con costo 0.25 bp, la migliore a -0.0057 bp al giorno, e un filtro sull'emivita stimata (al massimo 60 giorni) non cambia l'esito. Le sensibilità non servono a scegliere i parametri.

**H4, stabilità: confermata per le farfalle in 15 casi su 16.** Il segno del netto con costo 0.25 bp è lo stesso prima e dopo il 6 dicembre 2021: con lo scambio nello stesso giorno sette combinazioni su otto di farfalla e periodo sono negative e l'ottava vale +0.002 bp al giorno (t 0.07, indistinguibile da zero); con lo scambio il giorno dopo lo sono tutte e otto. Per H2 con scambio nello stesso giorno il netto è positivo in entrambi i periodi, con scambio il giorno dopo è negativo in entrambi (-0.0278 e -0.0072, quest'ultimo con t -0.88). Il dettaglio per periodo è in `output/h3_farfalle.csv` e `output/h2_residui.csv`.

Riepilogo:

| ipotesi | criterio | esito |
|---|---|---|
| H1 | Svensson riduce lo scarto più del solo rumore | in parte: solo a 20 e 30 anni (0.27 e 0.19 contro 0.55 e 0.48 del solo rumore), non fra 1 e 10 anni |
| H2 | netto a 0.25 bp positivo con t > 2 | criterio soddisfatto nello stesso giorno (t 8.4), non il giorno dopo (t -6.3); atteso anche da residui stazionari senza mispricing: nessuna prova di vantaggio |
| H3 | netto a 0.25 bp positivo con t > 2 per le farfalle, e la versione neutrale ai fattori fa meglio di quella DV01 | rifiutata, in entrambe le esecuzioni; la versione neutrale ai fattori è meglio in un solo caso su quattro |
| H4 | stesso segno prima e dopo dic 2021 | confermata per le farfalle in 15 casi su 16 (l'eccezione vale +0.002 bp, indistinguibile da zero) |

## 10. La curva di oggi

`scripts/analisi_corrente.py` scarica i giorni più recenti dopo la fine dello storico, adatta la curva e mostra per ogni scadenza il residuo, lo z-score sui 60 giorni precedenti e il percentile del |z| rispetto a tutto lo storico, più le farfalle. Il risultato dipende dal giorno in cui si esegue, quindi non è riportato qui. Uno z-score alto non è un segnale operativo: nello storico, il guadagno di questa regola sulle farfalle sparisce con costi fra 0.04 e 0.2 bp (sezione 9).

## 11. Limiti

- I punti del Tesoro sono letture interpolate, non rendimenti di un singolo titolo né prezzi eseguibili. Il loro rumore rientra da solo e gonfia i guadagni di un ritorno alla media, soprattutto con lo scambio nello stesso giorno del segnale. L'esecuzione "giorno dopo" è una prova grezza e non simula un'esecuzione reale. Il bias del dato gonfia i guadagni, quindi un risultato positivo va guardato con più sospetto di uno negativo.
- Il guadagno è calcolato sulla serie, non sulle obbligazioni: non include copertura reale, finanziamento né vincoli di esecuzione. I costi (0, 0.25, 0.5 bp per DV01 per gamba) sono ipotesi uniformi su tutte le scadenze, e il costo di H2 è su una gamba sola mentre coprire i fattori ne richiede almeno tre o quattro.
- Il confronto tra Tesoro e Fed mescola il premio del titolo nuovo con differenze di metodo, di orario (15:30 contro fine giornata) e di selezione dei titoli. Non separo queste cause, e non ho una spiegazione per il cambio di segno a 20 anni dopo il 2021.
- Non so a cosa attribuire il residuo di Nelson-Siegel: Svensson ne spiega una parte sistematica solo a 20 e 30 anni, e resta aperto se sia forma della curva, interpolazione del Tesoro, arrotondamento a 1 bp o liquidità.
- La stima non lineare ha minimi locali e parametri instabili (tau al limite in 464 giorni su 6502: 433 al limite superiore, 31 all'inferiore).
- La PCA, i fattori e la previsione trattano i par yield come rendimenti zero (con tau di Diebold-Li fisso): è un'approssimazione, mentre il fit giornaliero dei risultati reali usa i par yield. Il mondo sintetico usa invece tassi zero e minimi quadrati lineari su una griglia di 40 tau, quindi i falsi positivi con "tau libero" della sezione 7 non coincidono esattamente con la procedura dei dati reali.
- Il mondo sintetico ha errori indipendenti tra scadenze, rumore costante e parametri scelti da me; le soglie di potere valgono per quelle impostazioni e per il passo della griglia. Il controllo nullo della sezione 9 è un AR(1) per scadenza e non cattura la dipendenza fra scadenze.
- I p-value non sono corretti per test multipli. Con 8 scadenze, 4 farfalle, 3 costi e 2 esecuzioni, qualche |t| vicino a 2 può essere caso. I t di Newey-West con 250 ritardi sono prudenti, quelli con 10 ritardi più ottimisti: dove conta riporto più di una scelta. Il test di previsione ha poca potenza (periodo con tasso zero e shock del 2022, orizzonti sovrapposti).
- I parametri della sezione 9 sono convenzionali, ma non posso provare di averli fissati prima di vedere i dati.
- Il calcolo giornaliero della curva (circa 6500 giorni) viene salvato in `output/` e riusato dagli script: se si cambia il codice del fit bisogna cancellare `output/` per ricalcolarlo.
- Non uso prezzi di singole obbligazioni: dati gratuiti giornalieri per singolo titolo non esistono. La calibrazione sui prezzi con pesi 1/duration è nel pacchetto ma non nei risultati. Non ho esteso il lavoro all'euro.

## 12. Come riprodurre

Richiede Python 3.10 o superiore. Provato con Python 3.10 (pandas 2.3, numpy 2.2, scipy 1.15) e 3.13 (pandas 3.0, numpy 2.5, scipy 1.18).

```
pip install -e ".[dev,dati,notebook]"
pytest -q
ruff check --select E,F,B --ignore E501 src tests scripts
python scripts/scarica_storico.py      # solo se data/storico/ manca
python scripts/esegui_sintetico.py     # sezione 7 (circa 20 minuti)
python scripts/esegui_dati_reali.py    # sezione 8 (2-3 minuti)
python scripts/esegui_strategia.py     # sezione 9 (circa 5 minuti con i fit giornalieri già in output/, circa 30 senza)
python scripts/analisi_corrente.py     # sezione 10
python scripts/genera_grafici.py       # le immagini in img/
python scripts/genera_notebook.py      # il notebook
```

| Sezione | Script |
|---|---|
| 6 (controllo contro la Fed) | `tests/test_curva.py`, `notebooks/analisi.ipynb` (scarto massimo 0.00005) |
| 7 | `scripts/esegui_sintetico.py` |
| 8 | `scripts/esegui_dati_reali.py`; le autocorrelazioni a un giorno dei residui, in `scripts/esegui_strategia.py` e nel notebook |
| 9 | `scripts/esegui_strategia.py` |
| 10 | `scripts/analisi_corrente.py` |
| grafici | `scripts/genera_grafici.py` |

Gli script scrivono le tabelle in `output/` (non versionata). I test con simulazioni sono marcati `slow` (`pytest -m "not slow"` li salta; l'intera suite dura una decina di secondi). Lo storico in `data/storico/` è fisso: per rigenerarlo si cancella la cartella e si rilancia `scarica_storico.py`.

## 13. Bibliografia

- Nelson, C. R. e Siegel, A. F. (1987), Parsimonious modeling of yield curves, Journal of Business 60(4), 473-489.
- Svensson, L. E. O. (1994), Estimating and interpreting forward interest rates: Sweden 1992-1994, NBER Working Paper 4871.
- Diebold, F. X. e Li, C. (2006), Forecasting the term structure of government bond yields, Journal of Econometrics 130(2), 337-364.
- Gurkaynak, R. S., Sack, B. e Wright, J. H. (2007), The U.S. Treasury yield curve: 1961 to the present, Journal of Monetary Economics 54(8), 2291-2304.
- Newey, W. K. e West, K. D. (1987), A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix, Econometrica 55(3), 703-708.
- Diebold, F. X. e Mariano, R. S. (1995), Comparing predictive accuracy, Journal of Business and Economic Statistics 13(3), 253-263.
- Clark, T. E. e West, K. D. (2007), Approximately normal tests for equal predictive accuracy in nested models, Journal of Econometrics 138(1), 291-311.

"""Statistica di base per serie con sovrapposizioni o autocorrelazione."""
import numpy as np


def newey_west(x, ritardi: int) -> tuple[float, float]:
    """Media di una serie e statistica t con errore standard di Newey-West (pesi di Bartlett).

    Serve quando le osservazioni sono autocorrelate (guadagni giornalieri di posizioni che
    durano piu' giorni, finestre sovrapposte). Con ritardi = 0 e' il t ordinario.
    Restituisce (media, t).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    media = x.mean()
    d = x - media
    s = d @ d / n
    for k in range(1, ritardi + 1):
        s += 2 * (1 - k / (ritardi + 1)) * (d[k:] @ d[:-k]) / n
    return float(media), float(media / np.sqrt(s / n))

# -*- coding: utf-8 -*-

__all__ = ["get_power_spectrum"]

import numpy as np
from gatspy.periodic.lomb_scargle_fast import lomb_scargle_fast


def get_power_spectrum(t, f, fe, freq_uHz=None):
    if freq_uHz is None:
        freq_uHz = np.linspace(1, 300, 500000)  # uHz
    freq = freq_uHz*(24*60*60*1e-6)
    _, power = lomb_scargle_fast(t, f, dy=fe, f0=freq[0], df=freq[1]-freq[0],
                                 Nf=len(freq))
    return freq_uHz, power

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from acannon.data import load_light_curves
from acannon.spectrum import get_power_spectrum

OUTPUT_DIR = "spectra"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(row):
    kicid = int(row.kicid)
    t, f, fe = load_light_curves(kicid)
    freq, power = get_power_spectrum(t, f, fe)
    data = np.array(list(zip(freq, power)),
                    dtype=[("freq_uHz", float), ("power", float)])
    with h5py.File(os.path.join(OUTPUT_DIR, "{0}.h5".format(kicid)), "w") as f:
        f.attrs["kicid"] = kicid
        f.attrs["nu_max"] = float(row.nu_max)
        f.attrs["delta_nu"] = float(row.delta_nu)
        f.create_dataset("power", data=data)


df = pd.read_csv("stello_2013.dat", delim_whitespace=True,
                 names=["kicid", "nu_max", "delta_nu"])
pool = Pool()
list(tqdm(pool.imap_unordered(run, (
    row for _, row in df.iterrows()
)), total=len(df)))

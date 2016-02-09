# -*- coding: utf-8 -*-

__all__ = ["load_light_curves"]

import kplr
import numpy as np

from ._filter import running_median_trend


def load_light_curves(kicid, n_sigma_clip=10, sigma_clip=5.0, median_hw=10):
    client = kplr.API()
    lcs = client.light_curves(kicid, ktc_target_type="LC")

    t = []
    f = []
    fe = []
    for i, lc in enumerate(lcs):
        data, hdr = lc.read(header=True)
        x = data["TIME"]
        y = data["SAP_FLUX"]
        m = (data["SAP_QUALITY"] == 0) & np.isfinite(x) & np.isfinite(y)
        mu = np.median(y[m])
        x = x[m]
        y = (y[m] / mu - 1.0) * 1e6
        ye = data["SAP_FLUX_ERR"][m] * 1e6 / mu

        m0 = np.abs(y - np.mean(y)) < sigma_clip * np.std(y)
        for _ in range(n_sigma_clip):
            m = np.abs(y - np.mean(y[m0])) < sigma_clip * np.std(y[m0])
            if m.sum() == m0.sum():
                break
            m0[:] = m

        x = x[m]
        y = y[m]
        ye = ye[m]

        trend = running_median_trend(
            np.ascontiguousarray(x, dtype=float),
            np.ascontiguousarray(y, dtype=float),
            median_hw,
        )
        y -= trend

        m0 = np.abs(y - np.mean(y)) < sigma_clip * np.std(y)
        for _ in range(n_sigma_clip):
            m = np.abs(y - np.mean(y[m0])) < sigma_clip * np.std(y[m0])
            if m.sum() == m0.sum():
                break
            m0[:] = m

        t.append(x[m])
        f.append(y[m])
        fe.append(ye[m])

    t = np.concatenate(t)
    f = np.concatenate(f)
    fe = np.concatenate(fe)

    inds = np.argsort(t)
    t = np.ascontiguousarray(t[inds], dtype=float)
    f = np.ascontiguousarray(f[inds], dtype=float)
    fe = np.ascontiguousarray(fe[inds], dtype=float)

    return t, f, fe

"""Geodesy helpers.

The 90 trips are spread over the whole world (Europe, Russia, the Americas, Asia, Africa), so no
single UTM zone works. Each trip is projected with its own local azimuthal-equidistant CRS centred
on the trip's road-network bounding-box centre. Within one trip (GPS traces reach ~75 km across,
road networks ~140 km) the scale error of that projection is ~2e-5, i.e. ~2 cm per km, far below
GPS noise.

Never compute distances in raw degrees: 1 deg of longitude is ~111 km at the equator but only
~54 km at 61 deg N.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


class LocalProjector:
    """lon/lat (EPSG:4326, always lon first) <-> metres in a per-trip local plane."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    @classmethod
    def from_lonlat(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjector":
        return cls((np.min(lon) + np.max(lon)) / 2, (np.min(lat) + np.max(lat)) / 2)

    def to_xy(self, lon, lat):
        return self._fwd.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))

    def to_lonlat(self, x, y):
        return self._inv.transform(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))

import types

import numpy as np

from src.eval.metrics import haversine_km


def test_haversine_zero():
    d = haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert float(d[0]) == 0.0


def test_haversine_known():
    # NYC to LA ~ 3936 km
    nyc = (40.7128, -74.0060)
    la = (34.0522, -118.2437)
    d = haversine_km(np.array([nyc[0]]), np.array([nyc[1]]), np.array([la[0]]), np.array([la[1]]))
    assert 3800 < float(d[0]) < 4100
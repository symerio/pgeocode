# -*- coding: utf8 -*-
# License 3-clause BSD
#
# Authors: Roman Yurchak <roman.yurchak@symerio.com>
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

import pytest

import pgeocode
from pgeocode import haversine_distance, Nominatim, GeoDistance


@pytest.fixture
def temp_dir():
    path_save = pgeocode.STORAGE_DIR
    path = tempfile.mkdtemp()
    pgeocode.STORAGE_DIR = path
    yield path
    pgeocode.STORAGE_DIR = path_save
    shutil.rmtree(path)


def _normalize_str(x):
    if x is np.nan:
        return x
    else:
        return x.lower()


@pytest.mark.parametrize(
        'country, pc1, location1, pc2, location2, distance12',
        [('FR', '91120', 'Palaiseau', '67000', 'Strasbourg', 400),
         ('GB', 'WC2N 5DU', 'London', 'BT1 5GS', 'Belfast', 518),
         # ('AR', 'c1002', 'Buenos-Aires', '62091', 'Rio-Negro', 965), known failure   # noqa
         ('AU', '6837', 'Perth', '3000', 'melbourne', 2722),
         ('AU', '6837', 'Perth', '0221', 'Barton', 3089),
         ('US', '60605', 'Chicago', '94103', 'San Francisco', 2984),
         ('CA', 'M5R 1X8', 'Toronto', 'H2Z 1A7', 'Montreal', 503),
         ('IE', 'D01 R2PO', 'Dublin', 'T12 RW26', 'Cork', 219),
         ])
def test_countries(country, pc1, location1, pc2, location2,
                   distance12):
    if country == 'IE':
        pytest.xfail('TODO: Investigate failure for IE')
    nomi = Nominatim(country)

    res = nomi.query_postal_code(pc1)
    assert isinstance(res, pd.Series)
    assert _normalize_str(location1) in _normalize_str(res.place_name)

    res = nomi.query_postal_code(pc2)
    assert isinstance(res, pd.Series)
    assert _normalize_str(location2) in _normalize_str(res.place_name)

    gdist = GeoDistance(country)
    dist = gdist.query_postal_code(pc1, pc2)
    assert isinstance(dist, float)
    assert dist == pytest.approx(distance12, abs=5)


def test_download_dataset(temp_dir):
    assert not os.path.exists(os.path.join(temp_dir, 'FR.txt'))
    nomi = Nominatim('fr')
    # the data file was downloaded
    assert os.path.exists(os.path.join(temp_dir, 'FR.txt'))
    res = nomi.query_postal_code('77160')

    nomi2 = Nominatim('fr')
    res2 = nomi.query_postal_code('77160')

    assert_array_equal(nomi._data.columns,
                       nomi2._data.columns)
    assert_array_equal(nomi._data_frame.columns,
                       nomi2._data_frame.columns)
    assert nomi._data.shape == nomi._data.shape
    assert nomi._data_frame.shape == nomi._data_frame.shape

    assert len(res.place_name.split(',')) > 1
    assert len(res2.place_name.split(',')) > 1


def test_nominatim_query_postal_code():
    nomi = Nominatim('fr')

    res = nomi.query_postal_code(['91120'])
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == 1
    assert res.place_name.values[0] == 'Palaiseau'

    res = nomi.query_postal_code('91120')
    assert isinstance(res, pd.Series)
    assert res.place_name == 'Palaiseau'

    res = nomi.query_postal_code(['33625', '31000', '99999'])
    assert res.shape[0] == 3
    assert not np.isfinite(res.iloc[2].latitude)


def test_nominatim_query_postal_code_multiple():
    nomi = Nominatim('de', unique=False)
    expected_places = [
        'Wellen',
        'Groß Rodensleben',
        'Irxleben',
        'Eichenbarleben',
        'Klein Rodensleben',
        'Niederndodeleben',
        'Hohendodeleben',
        'Ochtmersleben',
    ]

    res = nomi.query_postal_code('39167')
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == len(expected_places)
    for place in res.place_name.values:
        assert place in expected_places


def test_nominatim_distance_postal_code():

    gdist = GeoDistance('fr')

    dist = gdist.query_postal_code('91120', '91120')
    assert dist == 0

    # distance between Palaiseau and Strasbourg
    dist = gdist.query_postal_code('91120', '67000')
    assert isinstance(dist, float)
    assert dist == pytest.approx(400, abs=4.5)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code('91120', ['31000', '67000'])
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code(['31000', '67000'], '91120')
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code(['31000', '67000'], ['67000', '31000'])
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.diff(dist)[0] == 0
    assert np.isfinite(dist).all()


def test_haversine_distance():
    try:
        from geopy.distance import great_circle
    except ImportError:
        raise pytest.skip('scikit-learn not installed')

    rng = np.random.RandomState(42)

    N = 100

    x = rng.rand(N, 2) * 80
    y = x * rng.rand(N, 2)

    d_ref = np.zeros(N)
    for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
        d_ref[idx] = great_circle(x_coord, y_coord).km

    d_pred = haversine_distance(x, y)
    # same distance +/- 3 km
    assert_allclose(d_ref, d_pred, atol=3)

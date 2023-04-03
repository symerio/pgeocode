# License 3-clause BSD
#
# Authors: Roman Yurchak <roman.yurchak@symerio.com>
import json
import os
import urllib
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pgeocode
from pgeocode import (
    GeoDistance,
    Nominatim,
    _open_extract_url,
    haversine_distance,
)


@pytest.fixture
def temp_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(pgeocode, "STORAGE_DIR", str(tmpdir))
    yield str(tmpdir)


def _normalize_str(x):
    if x is np.nan:
        return x
    else:
        return x.lower()


@pytest.mark.parametrize(
    "country, pc1, location1, pc2, location2, distance12",
    [
        ("FR", "91120", "Palaiseau", "67000", "Strasbourg", 400),
        ("GB", "WC2N 5DU", "London", "BT1 5GS", "Belfast", 518),
        # ('AR', 'c1002', 'Buenos-Aires', '62091', 'Rio-Negro', 965), known failure   # noqa
        ("AU", "6837", "Perth", "3000", "melbourne", 2722),
        ("AU", "6837", "Perth", "0221", "Barton", 3089),
        ("US", "60605", "Chicago", "94103", "San Francisco", 2984),
        ("CA", "M5R 1X8", "Toronto", "H2Z 1A7", "Montreal", 503),
        ("IE", "D01 R2PO", "Dublin", "T12 RW26", "Cork", 219),
    ],
)
def test_countries(country, pc1, location1, pc2, location2, distance12):
    if country == "IE":
        pytest.xfail("TODO: Investigate failure for IE")
    nomi = Nominatim(country)

    res = nomi.query_postal_code(pc1)
    assert isinstance(res, pd.Series)
    assert _normalize_str(location1) in _normalize_str(res.place_name)

    assert "country_code" in res.index

    res = nomi.query_postal_code(pc2)
    assert isinstance(res, pd.Series)
    assert _normalize_str(location2) in _normalize_str(res.place_name)

    gdist = GeoDistance(country)
    dist = gdist.query_postal_code(pc1, pc2)
    assert isinstance(dist, float)
    assert dist == pytest.approx(distance12, abs=5)


def test_download_dataset(temp_dir):
    assert not os.path.exists(os.path.join(temp_dir, "FR.txt"))
    nomi = Nominatim("fr")
    # the data file was downloaded
    assert os.path.exists(os.path.join(temp_dir, "FR.txt"))
    res = nomi.query_postal_code("77160")

    nomi2 = Nominatim("fr")
    res2 = nomi.query_postal_code("77160")

    assert_array_equal(nomi._data.columns, nomi2._data.columns)
    assert_array_equal(nomi._data_frame.columns, nomi2._data_frame.columns)
    assert nomi._data.shape == nomi._data.shape
    assert nomi._data_frame.shape == nomi._data_frame.shape

    assert len(res.place_name.split(",")) > 1
    assert len(res2.place_name.split(",")) > 1


def test_nominatim_query_postal_code():
    nomi = Nominatim("fr")

    res = nomi.query_postal_code(["91120"])
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == 1
    assert res.place_name.values[0] == "Palaiseau"

    res = nomi.query_postal_code("91120")
    assert isinstance(res, pd.Series)
    assert res.place_name == "Palaiseau"

    res = nomi.query_postal_code(["33625", "31000", "99999"])
    assert res.shape[0] == 3
    assert not np.isfinite(res.iloc[2].latitude)


def test_nominatim_query_postal_code_multiple():
    nomi = Nominatim("de", unique=False)
    expected_places = [
        "Wellen",
        "Groß Rodensleben",
        "Irxleben",
        "Eichenbarleben",
        "Klein Rodensleben",
        "Niederndodeleben",
        "Hohendodeleben",
        "Ochtmersleben",
    ]

    res = nomi.query_postal_code("39167")
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == len(expected_places)
    for place in res.place_name.values:
        assert place in expected_places


@pytest.mark.slow
@pytest.mark.parametrize("country", pgeocode.COUNTRIES_VALID)
def test_nominatim_all_countries(country):
    nomi = Nominatim(country)
    res = nomi.query_postal_code("00000")
    assert isinstance(res, pd.Series)


def test_nominatim_distance_postal_code():
    gdist = GeoDistance("fr")

    dist = gdist.query_postal_code("91120", "91120")
    assert dist == 0

    # distance between Palaiseau and Strasbourg
    dist = gdist.query_postal_code("91120", "67000")
    assert isinstance(dist, float)
    assert dist == pytest.approx(400, abs=4.5)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code("91120", ["31000", "67000"])
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code(["31000", "67000"], "91120")
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.isfinite(dist).all()

    dist = gdist.query_postal_code(["31000", "67000"], ["67000", "31000"])
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.diff(dist)[0] == 0
    assert np.isfinite(dist).all()


def test_haversine_distance():
    try:
        from geopy.distance import great_circle
    except ImportError:
        raise pytest.skip("scikit-learn not installed")

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


def test_open_extract_url(httpserver):
    download_url = "/fr.txt"

    # check download of uncompressed files
    httpserver.expect_oneshot_request(download_url).respond_with_json({"a": 1})
    with _open_extract_url(httpserver.url_for(download_url), "fr") as fh:
        assert json.loads(fh.read()) == {"a": 1}
    httpserver.check_assertions()

    # check download of zipped files
    # Create an in-memory zip file
    answer = b"a=1"
    with BytesIO() as fh:
        with ZipFile(fh, "w") as fh_zip:
            with fh_zip.open("FR.txt", "w") as fh_inner:
                fh_inner.write(answer)
        fh.seek(0)
        res = fh.read()

    download_url = "/fr.zip"
    httpserver.expect_oneshot_request(download_url).respond_with_data(res)

    with _open_extract_url(httpserver.url_for(download_url), "fr") as fh:
        assert fh.read() == answer


@pytest.mark.parametrize(
    "download_url",
    [
        "https://download.geonames.org/export/zip/{country}.zip",
        "https://symerio.github.io/postal-codes-data/data/"
        "geonames/{country}.txt",
    ],
    ids=["geonames", "gitlab-pages"],
)
def test_cdn(temp_dir, monkeypatch, download_url):
    monkeypatch.setattr(pgeocode, "DOWNLOAD_URL", [download_url])
    assert not os.path.exists(os.path.join(temp_dir, "IE.txt"))
    Nominatim("IE")
    # the data file was downloaded
    assert os.path.exists(os.path.join(temp_dir, "IE.txt"))


def test_url_returns_404(httpserver, monkeypatch, temp_dir):
    download_url = "/fr.gzip"
    httpserver.expect_oneshot_request(download_url).respond_with_data(
        "", status=404
    )

    monkeypatch.setattr(
        pgeocode, "DOWNLOAD_URL", [httpserver.url_for(download_url)]
    )
    # Nominatim("fr")
    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 404"):
        Nominatim("fr")
    httpserver.check_assertions()


def test_first_url_fails(httpserver, monkeypatch, temp_dir):
    download_url = "/IE.txt"
    httpserver.expect_oneshot_request(download_url).respond_with_data(
        "", status=404
    )

    monkeypatch.setattr(
        pgeocode,
        "DOWNLOAD_URL",
        [
            httpserver.url_for(download_url),
            "https://symerio.github.io/postal-codes-data/data/"
            "geonames/{country}.txt",
        ],
    )
    msg = "IE.txt failed with: HTTP Error 404.*Trying next URL"
    with pytest.warns(UserWarning, match=msg):
        Nominatim("ie")
    httpserver.check_assertions()


def test_query_location_exact():
    nomi = Nominatim("fr")
    res = nomi.query_location("Strasbourg")
    assert isinstance(res, pd.DataFrame)

    # Invalid query
    res = nomi.query_location("182581stisdgsg21191t..,,,,,,,,,,")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0

    # Query on a different field name
    res = nomi.query_location("île", col="state_name")
    assert isinstance(res, pd.DataFrame)
    assert res["state_name"].unique().tolist() == ["Île-de-France"]


def test_query_location_fuzzy():
    pytest.importorskip("thefuzz")
    nomi = Nominatim("fr")
    # Querying with a typo
    res = nomi.query_location("Straasborg", fuzzy_threshold=80)
    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    assert res["place_name"].unique().tolist() == ["Strasbourg"]


def test_unique_index_pcode(tmp_path):
    """Check that a centroid is computed both for latitude and longitude

    Regression test for https://github.com/symerio/pgeocode/pull/62
    """

    class MockNominatim(Nominatim):
        def __init__(self):
            pass

    data = pd.DataFrame(
        {
            "postal_code": ["1", "1", "2", "2"],
            "latitude": [1.0, 2.0, 3.0, 4],
            "longitude": [5.0, 6.0, 7.0, 8],
            "place_name": ["a", "b", "c", "d"],
            "state_name": ["a", "b", "c", "d"],
            "country_name": ["a", "b", "c", "d"],
            "county_name": ["a", "b", "c", "d"],
            "community_name": ["a", "b", "c", "d"],
            "accuracy": [1, 2, 3, 4],
            "country_code": [1, 2, 3, 4],
            "county_code": [1, 2, 3, 4],
            "state_code": [1, 2, 3, 4],
            "community_code": [1, 2, 3, 4],
        }
    )

    nominatim = MockNominatim()
    data_path = tmp_path / "a.txt"
    nominatim._data_path = str(data_path)
    nominatim._data = data
    data_unique = nominatim._index_postal_codes()

    data_unique_expected = pd.DataFrame(
        {
            "postal_code": ["1", "2"],
            "latitude": [1.5, 3.5],
            "longitude": [5.5, 7.5],
            "place_name": ["a, b", "c, d"],
            "state_name": ["a", "c"],
            # We don't include the country_name for some reason?
            # 'country_name': ['a', 'c'],
            "county_name": ["a", "c"],
            "community_name": ["a", "c"],
            "accuracy": [1, 3],
            "country_code": [1, 3],
            "county_code": [1, 3],
            "state_code": [1, 3],
            "community_code": [1, 3],
        }
    )
    pd.testing.assert_frame_equal(
        data_unique.sort_index(axis=1), data_unique_expected.sort_index(axis=1)
    )

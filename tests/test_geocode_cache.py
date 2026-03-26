from types import SimpleNamespace

from src.data.geocode import GeoNamesClient


def test_geocode_cache(monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=15, verify=True):
        data = {"geonames": [{"name": "TestCity", "adminName1": "TestState", "countryCode": "US", "lat": "1.0", "lng": "2.0"}]}
        return SimpleNamespace(status_code=200, json=lambda: data)

    monkeypatch.setattr("requests.get", fake_get)
    client = GeoNamesClient(username="dummy", cache_path=tmp_path / "geo.sqlite", rate_limit=1000)
    res1 = client.reverse_geocode(1.0, 2.0)
    res2 = client.reverse_geocode(1.0, 2.0)
    assert res1.name == "TestCity"
    assert res2.name == "TestCity"

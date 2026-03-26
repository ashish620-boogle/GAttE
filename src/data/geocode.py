from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class GeoResult:
    name: str
    admin1: str
    country: str
    lat: float
    lon: float


class GeoNamesClient:
    def __init__(
        self,
        username: str,
        cache_path: str | Path,
        rate_limit: float = 1.0,
        round_precision: int = 4,
        country_code: str = "US",
        verify_ssl: bool = True,
        allow_insecure_fallback: bool = False,
    ) -> None:
        if not username:
            raise ValueError("GeoNames username is required (set GEONAMES_USERNAME)")
        self.username = username
        self.rate_limit = max(rate_limit, 0.1)
        self.round_precision = round_precision
        self.country_code = country_code
        self.verify_ssl = verify_ssl
        self.allow_insecure_fallback = allow_insecure_fallback
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path)
        self._last_call = 0.0
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reverse_cache (
                lat REAL,
                lon REAL,
                name TEXT,
                admin1 TEXT,
                country TEXT,
                PRIMARY KEY (lat, lon)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_cache (
                name TEXT,
                admin1 TEXT,
                country TEXT,
                lat REAL,
                lon REAL,
                PRIMARY KEY (name, admin1, country)
            )
            """
        )
        self._conn.commit()

    def _sleep_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _round(self, value: float) -> float:
        return round(float(value), self.round_precision)

    def reverse_geocode(self, lat: float, lon: float) -> Optional[GeoResult]:
        rlat = self._round(lat)
        rlon = self._round(lon)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name, admin1, country, lat, lon FROM reverse_cache WHERE lat=? AND lon=?",
            (rlat, rlon),
        )
        row = cur.fetchone()
        if row:
            return GeoResult(name=row[0], admin1=row[1], country=row[2], lat=row[3], lon=row[4])

        self._sleep_rate_limit()
        params = {
            "lat": rlat,
            "lng": rlon,
            "username": self.username,
        }
        url = "https://api.geonames.org/findNearbyPlaceNameJSON"
        try:
            resp = requests.get(url, params=params, timeout=15, verify=self.verify_ssl)
        except requests.exceptions.SSLError:
            if not self.allow_insecure_fallback:
                raise
            resp = requests.get(url, params=params, timeout=15, verify=False)
        self._last_call = time.time()
        if resp.status_code != 200:
            return None
        data = resp.json()
        geonames = data.get("geonames", [])
        if not geonames:
            return None
        g = geonames[0]
        result = GeoResult(
            name=g.get("name", ""),
            admin1=g.get("adminName1", ""),
            country=g.get("countryCode", ""),
            lat=float(g.get("lat", rlat)),
            lon=float(g.get("lng", rlon)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO reverse_cache (lat, lon, name, admin1, country) VALUES (?,?,?,?,?)",
            (rlat, rlon, result.name, result.admin1, result.country),
        )
        self._conn.commit()
        return result

    def forward_geocode(self, name: str, admin1: str = "", country: Optional[str] = None) -> Optional[GeoResult]:
        name_key = name.strip().lower()
        admin1_key = (admin1 or "").strip().lower()
        country_key = (country or self.country_code).strip().upper()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name, admin1, country, lat, lon FROM forward_cache WHERE name=? AND admin1=? AND country=?",
            (name_key, admin1_key, country_key),
        )
        row = cur.fetchone()
        if row:
            return GeoResult(name=row[0], admin1=row[1], country=row[2], lat=row[3], lon=row[4])

        self._sleep_rate_limit()
        params = {
            "name": name,
            "maxRows": 1,
            "country": country_key,
            "username": self.username,
        }
        if admin1:
            params["adminName1"] = admin1
        url = "https://api.geonames.org/searchJSON"
        try:
            resp = requests.get(url, params=params, timeout=15, verify=self.verify_ssl)
        except requests.exceptions.SSLError:
            if not self.allow_insecure_fallback:
                raise
            resp = requests.get(url, params=params, timeout=15, verify=False)
        self._last_call = time.time()
        if resp.status_code != 200:
            return None
        data = resp.json()
        geonames = data.get("geonames", [])
        if not geonames:
            return None
        g = geonames[0]
        result = GeoResult(
            name=g.get("name", ""),
            admin1=g.get("adminName1", ""),
            country=g.get("countryCode", ""),
            lat=float(g.get("lat", 0.0)),
            lon=float(g.get("lng", 0.0)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO forward_cache (name, admin1, country, lat, lon) VALUES (?,?,?,?,?)",
            (name_key, admin1_key, country_key, result.lat, result.lon),
        )
        self._conn.commit()
        return result


def get_geonames_username(env_name: str = "GEONAMES_USERNAME") -> str:
    value = os.getenv(env_name, "").strip()
    return value or "ashishboogle810"

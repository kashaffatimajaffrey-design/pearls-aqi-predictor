"""Feature Store abstraction.

Two interchangeable backends behind one interface:

* HopsworksFeatureStore -- the real thing. Upserts into a versioned Feature
  Group keyed on `ts`, with `ts` as the event time so backfill and hourly
  inserts de-duplicate correctly.
* LocalFeatureStore     -- append-and-dedupe Parquet on disk. This is what runs
  in CI and on a fresh clone; the committed Parquet file is also what makes the
  Streamlit dashboard live without a Hopsworks account.

The pipelines only ever talk to this interface, so switching backends is a
single environment variable (HOPSWORKS_API_KEY / HOPSWORKS_PROJECT).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .. import config

log = logging.getLogger(__name__)

PRIMARY_KEY = "ts"


class BaseFeatureStore(ABC):
    name: str

    @abstractmethod
    def write(self, df: pd.DataFrame, group: str = config.FG_NAME) -> int:
        """Upsert rows. Returns the number of rows written."""

    @abstractmethod
    def read(self, group: str = config.FG_NAME) -> pd.DataFrame:
        """Read the whole feature group back, sorted by ts."""

    def read_latest(self, n: int = 1, group: str = config.FG_NAME) -> pd.DataFrame:
        df = self.read(group)
        return df.tail(n).reset_index(drop=True) if not df.empty else df


class LocalFeatureStore(BaseFeatureStore):
    """Parquet-backed store. Same upsert-on-primary-key semantics as Hopsworks."""

    name = "local"

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.FEATURE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, group: str) -> Path:
        return self.root / f"{group}_v{config.FG_VERSION}.parquet"

    def write(self, df: pd.DataFrame, group: str = config.FG_NAME) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df[PRIMARY_KEY] = pd.to_datetime(df[PRIMARY_KEY], utc=True)

        path = self._path(group)
        if path.exists():
            existing = pd.read_parquet(path)
            existing[PRIMARY_KEY] = pd.to_datetime(existing[PRIMARY_KEY], utc=True)
            # New rows win on conflict -- a later run has fresher upstream data.
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=PRIMARY_KEY, keep="last")
        else:
            combined = df

        combined = combined.sort_values(PRIMARY_KEY).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        log.info("local feature store: %d rows total at %s", len(combined), path)
        return len(df)

    def read(self, group: str = config.FG_NAME) -> pd.DataFrame:
        path = self._path(group)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df[PRIMARY_KEY] = pd.to_datetime(df[PRIMARY_KEY], utc=True)
        return df.sort_values(PRIMARY_KEY).reset_index(drop=True)


class HopsworksFeatureStore(BaseFeatureStore):
    """Hopsworks Feature Group backend."""

    name = "hopsworks"

    def __init__(self):
        import hopsworks  # imported lazily -- heavy, and optional

        self.project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT or None,
        )
        self.fs = self.project.get_feature_store()
        self._mirror = LocalFeatureStore()  # keeps the dashboard usable offline
        log.info("connected to Hopsworks project %s", self.project.name)

    def _feature_group(self, group: str, df: pd.DataFrame | None = None):
        try:
            fg = self.fs.get_feature_group(name=group, version=config.FG_VERSION)
            if fg is not None:
                return fg
        except Exception:
            pass
        if df is None:
            return None
        return self.fs.get_or_create_feature_group(
            name=group,
            version=config.FG_VERSION,
            primary_key=[PRIMARY_KEY],
            event_time=PRIMARY_KEY,
            description=f"Hourly engineered AQI features for {config.CITY}",
            online_enabled=False,
        )

    @staticmethod
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """Hopsworks requires lowercase, underscore-only feature names."""
        out = df.copy()
        out.columns = [c.lower().replace("-", "_").replace(" ", "_") for c in out.columns]
        out[PRIMARY_KEY] = pd.to_datetime(out[PRIMARY_KEY], utc=True)
        return out

    def write(self, df: pd.DataFrame, group: str = config.FG_NAME) -> int:
        if df is None or df.empty:
            return 0
        payload = self._sanitize(df)
        fg = self._feature_group(group, payload)
        fg.insert(payload, write_options={"wait_for_job": False})
        self._mirror.write(df, group)
        log.info("hopsworks: inserted %d rows into %s", len(payload), group)
        return len(payload)

    def read(self, group: str = config.FG_NAME) -> pd.DataFrame:
        fg = self._feature_group(group)
        if fg is None:
            return self._mirror.read(group)
        try:
            df = fg.read()
        except Exception as exc:  # pragma: no cover - network
            log.warning("hopsworks read failed (%s); using local mirror", exc)
            return self._mirror.read(group)
        if df is None or df.empty:
            return self._mirror.read(group)
        df[PRIMARY_KEY] = pd.to_datetime(df[PRIMARY_KEY], utc=True)
        return df.sort_values(PRIMARY_KEY).reset_index(drop=True)


def get_feature_store(kind: str | None = None) -> BaseFeatureStore:
    """Resolve the configured backend, degrading to local if Hopsworks is
    unreachable so a credential problem never takes the pipeline down."""
    kind = (kind or config.resolved_store()).lower()
    if kind == "hopsworks":
        try:
            return HopsworksFeatureStore()
        except Exception as exc:
            log.warning("Hopsworks unavailable (%s); falling back to local store", exc)
    return LocalFeatureStore()

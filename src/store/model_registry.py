"""Model Registry abstraction.

Mirrors the Feature Store split: a Hopsworks-backed registry when credentials
exist, otherwise a versioned directory on disk. Both expose the same three
operations -- push a model bundle, resolve the current production model, and
list the version history.

A "bundle" is a directory containing the serialised estimator plus a
`metadata.json` describing the training run (metrics, feature list, data window,
git sha). The dashboard and the Flask API both load through this interface.
"""
from __future__ import annotations

import json
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

PRODUCTION_LINK = "production"


class BaseModelRegistry(ABC):
    name: str

    @abstractmethod
    def push(self, bundle_dir: Path, metadata: dict) -> str:
        """Register a bundle. Returns the version identifier."""

    @abstractmethod
    def production_path(self) -> Path | None:
        """Local path to the current production bundle, or None."""

    @abstractmethod
    def history(self) -> list[dict]:
        """Metadata for every registered version, newest first."""


class LocalModelRegistry(BaseModelRegistry):
    name = "local"

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.REGISTRY_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _version_dirs(self) -> list[Path]:
        return sorted(
            (p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v")),
            key=lambda p: p.name,
        )

    def next_version(self) -> str:
        existing = self._version_dirs()
        n = 0
        for p in existing:
            try:
                n = max(n, int(p.name.lstrip("v").split("_")[0]))
            except ValueError:
                continue
        return f"v{n + 1:04d}"

    def push(self, bundle_dir: Path, metadata: dict) -> str:
        version = self.next_version()
        target = self.root / version
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle_dir, target)

        metadata = {
            **metadata,
            "version": version,
            "registered_at": datetime.now(UTC).isoformat(),
            "registry": self.name,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # `production` is a plain copy rather than a symlink: Windows and the
        # GitHub Actions checkout both handle copies predictably.
        prod = self.root / PRODUCTION_LINK
        if prod.exists():
            shutil.rmtree(prod)
        shutil.copytree(target, prod)

        log.info("registered model %s (production updated)", version)
        return version

    def production_path(self) -> Path | None:
        prod = self.root / PRODUCTION_LINK
        if prod.exists():
            return prod
        versions = self._version_dirs()
        return versions[-1] if versions else None

    def history(self) -> list[dict]:
        out = []
        for p in self._version_dirs():
            meta = p / "metadata.json"
            if meta.exists():
                try:
                    out.append(json.loads(meta.read_text()))
                except json.JSONDecodeError:
                    continue
        return sorted(out, key=lambda m: m.get("registered_at", ""), reverse=True)


class HopsworksModelRegistry(BaseModelRegistry):
    """Hopsworks Model Registry, with a local mirror so the dashboard can always
    load a model without a round trip."""

    name = "hopsworks"

    def __init__(self):
        import hopsworks

        self.project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT or None,
        )
        self.mr = self.project.get_model_registry()
        self._mirror = LocalModelRegistry()

    def push(self, bundle_dir: Path, metadata: dict) -> str:
        version = self._mirror.push(bundle_dir, {**metadata, "registry": self.name})
        metrics = {
            k: float(v)
            for k, v in (metadata.get("metrics") or {}).items()
            if isinstance(v, (int, float))
        }
        try:
            model = self.mr.python.create_model(
                name=config.MODEL_REGISTRY_NAME,
                metrics=metrics,
                description=(
                    f"{metadata.get('best_model', 'model')} forecasting AQI at "
                    f"+24/48/72h for {config.CITY}"
                ),
            )
            model.save(str(bundle_dir))
            log.info("pushed model to Hopsworks registry as %s", config.MODEL_REGISTRY_NAME)
        except Exception as exc:  # pragma: no cover - network
            log.warning("Hopsworks model push failed (%s); local copy retained", exc)
        return version

    def production_path(self) -> Path | None:
        local = self._mirror.production_path()
        if local is not None:
            return local
        try:  # pragma: no cover - network
            model = self.mr.get_best_model(
                config.MODEL_REGISTRY_NAME, "rmse", "min"
            )
            return Path(model.download())
        except Exception as exc:
            log.warning("could not pull model from Hopsworks: %s", exc)
            return None

    def history(self) -> list[dict]:
        return self._mirror.history()


def get_model_registry(kind: str | None = None) -> BaseModelRegistry:
    kind = (kind or config.resolved_store()).lower()
    if kind == "hopsworks":
        try:
            return HopsworksModelRegistry()
        except Exception as exc:
            log.warning("Hopsworks registry unavailable (%s); using local", exc)
    return LocalModelRegistry()

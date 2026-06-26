"""Configuration for the heracleum-tox MCP server (pydantic-settings).

Values are read from environment / .env with the ``HERACLEUM_`` prefix, mirroring
the other CoScientist MCP servers (``TOX_``, ``CHEM_`` ...).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_DATA_FILE = PACKAGE_DIR / "data" / "heracleum_metabolites.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DEFAULT_MODEL_CACHE = PACKAGE_DIR / "model_cache"


class Settings(BaseSettings):
    """Server settings."""

    model_config = SettingsConfigDict(
        env_prefix="HERACLEUM_",
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- MCP transport (matches the other CoScientist MCP servers) ---
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=7331)
    mcp_path: str = Field(default="/mcp")

    # --- dataset ---
    dataset_path: str = Field(default=str(DEFAULT_DATA_FILE))

    # --- chemical space / clustering (open-source analogue of SynMap) ---
    morgan_radius: int = Field(default=2)        # ECFP4
    morgan_nbits: int = Field(default=2048)
    n_clusters: int = Field(default=5)           # paper: five clusters A-E
    tsne_perplexity: float = Field(default=15.0)
    random_state: int = Field(default=42)

    # --- ML models (open-source analogue of Syntelly CatBoost/XGBoost) ---
    model_cache_dir: str = Field(default=str(DEFAULT_MODEL_CACHE))
    model_backend: str = Field(default="catboost")   # "catboost" | "xgboost"
    retrain: bool = Field(default=False)             # force re-training, ignore cache

    # --- applicability domain (paper section 2.5: kNN k=5 + Gaussian) ---
    ad_k_neighbors: int = Field(default=5)
    # Per-endpoint AD distance thresholds are learnt from the training set
    # (mean + ad_threshold_sigma * std of the in-training NN distances).
    ad_threshold_sigma: float = Field(default=2.0)

    # --- synthesis cost (open-source analogue of Syntelly Synthesis cost) ---
    # Optional ASKCOS-backed retrosynthesis service (reuse chemical-mcp-server's).
    askcos_url: str = Field(default="")
    synthesis_max_steps: int = Field(default=6)      # paper: 1-6 stages

    # --- artifacts (figures): local dir or S3-compatible bucket ---
    artifacts_dir: str = Field(default=str(DEFAULT_ARTIFACTS_DIR))
    artifact_url_base: str = Field(default="")
    s3_endpoint_url: str = Field(default="")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_bucket_name: str = Field(default="")
    s3_url_expiration: int = Field(default=3600)

    @property
    def use_s3(self) -> bool:
        return bool(self.s3_endpoint_url and self.s3_bucket_name and self.s3_access_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()

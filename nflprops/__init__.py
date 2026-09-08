"""NFL player prop projection system."""
from .config import DEFAULT_CONFIG, ModelConfig
from .data.loaders import DataStore
from .projections import ProjectionEngine
from .output import JSON_SCHEMA, write_json

__version__ = "1.0.0"
__all__ = ["DEFAULT_CONFIG", "ModelConfig", "DataStore", "ProjectionEngine",
           "JSON_SCHEMA", "write_json"]

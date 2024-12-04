# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.35"

# from ultralytics.data.explorer.explorer import Explorer
from .models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from .models.fastsam import FastSAM
from .models.nas import NAS
from .utils import ASSETS, SETTINGS as settings
from .utils.checks import check_yolo as checks
from .utils.downloads import download
# from ultralytics.nn.tasks import FeatureExtraction

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10"
    # "FeatureExtraction",
)

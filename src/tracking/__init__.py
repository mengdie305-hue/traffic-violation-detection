from .deepsort_tracker import DeepSORTTracker
from .realtime_deepsort_tracker import RealtimeDeepSORTTracker


def create_tracker(backend: str = "simple", **kwargs):
    """创建统一接口的追踪器。"""
    if backend == "simple":
        allowed = {"max_age", "min_hits", "iou_threshold", "max_distance"}
        return DeepSORTTracker(**{
            key: value for key, value in kwargs.items() if key in allowed
        })
    if backend == "realtime_deepsort":
        allowed = {
            "max_age", "min_hits", "max_iou_distance",
            "max_cosine_distance", "embedder", "embedder_gpu",
        }
        return RealtimeDeepSORTTracker(**{
            key: value for key, value in kwargs.items() if key in allowed
        })
    raise ValueError(f"Unsupported tracking backend: {backend}")

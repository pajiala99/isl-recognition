"""Indian Sign Language recognition on keypoint sequences.

MediaPipe's native layers (glog, TFLite, XNNPACK) print startup noise to stderr the
moment they load, which looks like errors during a live demo. These defaults quieten
them and must be set before mediapipe is imported anywhere - which this package
__init__ guarantees. Export any of them yourself to get the logs back.
"""
import os as _os

for _k, _v in (("GLOG_minloglevel", "2"),        # 0=INFO 1=WARNING 2=ERROR 3=FATAL
               ("GLOG_stderrthreshold", "3"),
               ("GLOG_logtostderr", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"),
               ("ABSL_MIN_LOG_LEVEL", "3")):
    _os.environ.setdefault(_k, _v)

__version__ = "0.1.0"

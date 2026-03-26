import os

# Silence TensorFlow info/warnings and avoid oneDNN banner.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Prefer legacy Keras for TensorFlow Hub compatibility (Keras 3).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

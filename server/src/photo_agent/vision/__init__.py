"""AI models for edits that understand what is in the photo.

Models run in a separate worker process (`worker.py`) that takes jobs from a queue, reports
progress, and keeps loaded models and downloaded weights cached. Each task (find the sky,
fill in a removed object, ...) has an open-source model backend and a classical computer
vision fallback (`backends.py`), so the app works, more roughly, without a GPU or weights.
"""

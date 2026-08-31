"""
SatQuery AI — Service Layer (future).

This package will contain business logic that sits between the API routes
and the ML model wrappers, handling:
  - Image preprocessing (resize, normalize, SAR speckle filtering)
  - Result post-processing (confidence calibration, evidence assembly)
  - Report generation
  - Caching / batching
"""

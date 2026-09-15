"""ecg_core: shared core package for the ECG real-time analytics framework.

Provides configuration, feature definitions, preprocessing, model
definitions, and inference logic used by training, streaming, and API
layers. Keeping this logic in one package guarantees that training,
Spark, and the FastAPI service all agree on feature ordering, label
encoding, and thresholds.
"""

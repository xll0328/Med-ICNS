# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
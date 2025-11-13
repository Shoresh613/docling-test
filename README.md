# Docling with VLM
This script is for simple evaluation of different providers and models for VLM interpretation of images in PDF files. 
## Usage
Either set provider in env var `VISION_BACKEND` to one of ollama, openai, openrouter or gemini (gemini code not tested and probably not working 100%), or choose provider by altering the row, replacing ollama with the desired provider:

```python
PROVIDER = os.getenv("VISION_BACKEND", "ollama")
```

The models for the different providers can for now not be set using env vars, for that you need to alter the code. This can be done in the PROVIDERS dict around line 39.

When all is set, simply run it with `python docling_vlm.py` (or `time python docling_vlm.py` on Linux to measure how long it took), installing missing requirements (they are not included in the `pyproject.toml` file as this is still a test script external to the main application).
When finished, an `output.md` file is generated for evaluation.

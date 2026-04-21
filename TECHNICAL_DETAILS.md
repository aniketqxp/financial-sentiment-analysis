# Technical Implementation Manual

This document provides a granular audit of the engineering decisions, algorithmic implementations, and architectural trade-offs within the Financial Sentiment Analysis project.

## 1. Logic & Algorithms

### OCR Pipeline & Pre-processing
The system utilizes `pytesseract` for text extraction. To mitigate common OCR degradation issues (e.g., low contrast in financial reports), a 1.5x contrast enhancement kernel is applied via `PIL.ImageEnhance.Contrast` before character recognition.
- **Metadata Extraction**: The pipeline captures average confidence scores and low-confidence word counts (threshold < 60) to provide data quality metrics to the user.

### LLM-Based Structural Refinement
Large Language Models (Llama 3.1 8B via Hugging Face Inference API) are used not for sentiment, but for **structural correction**. 
- **The Why**: OCR outputs are often fragmented or contains "noise" (e.g., misread symbols). The LLM is prompted to perform deterministic sentence segmentation and spelling correction, returning a structured JSON array for downstream processing.

### Sentiment Engine: FinBERT
Instead of generic sentiment models, the project employs `ProsusAI/finbert`, a BERT model pre-trained on the Financial PhraseBank dataset.
- **Statistical Nuance**: FinBERT handles the nuance of financial jargon (e.g., "growth" is positive, but "growth in liabilities" is negative) better than general-purpose models.
- **Logit Processing**: Softmax is applied to model outputs to generate probability distributions across Positive, Negative, and Neutral classes.

### Local Interpretable Model-agnostic Explanations (LIME)
Explainability is implemented via `LimeTextExplainer`.
- **Mathematical Transformation**: LIME perturbs input text by removing words and observing the change in the model's prediction. A linear model is then fit locally to these perturbations to calculate "feature importance" (word contribution).
- **Custom Wrapper**: A `predictor_fn` was implemented to bridge the gap between LIME's expectation of raw text and FinBERT's requirement for tokenization and tensor conversion.

## 2. Engineering "Fires" & Hacks

### Module Resolution & sys.path Injection
To support a modular `src/` directory while allowing standalone execution of scripts in `scripts/`, a runtime `sys.path` injection was implemented in `generate_materials.py`:
```python
sys.path.append(str(Path(__file__).parent.parent))
```
- **The Why**: This resolves `ModuleNotFoundError` when the project is not installed in editable mode, ensuring tools can find the core logic relative to their own location.

### Path Portability & Sanitization
The project underwent a sanitation pass to remove absolute Windows-specific paths (e.g., `D:/Desktop/...`). These were replaced with relative `Path` objects from `pathlib` to ensure cross-platform compatibility (Windows/Linux/MacOS).

### Environment Configuration Fallbacks
The `.env` loader is initialized at the top level of both the entry point and core logic. This ensures that Hugging Face and Google GenAI tokens are accessible regardless of the execution context.

## 3. Performance & Scaling

### API Rate Limiting & Resilience
A custom `RateLimiter` class manages API throughput to prevent 429 errors from the Hugging Face Inference API.
- **Backoff Strategy**: The `retry_with_backoff` decorator implements exponential backoff, doubling the wait time after each failure (initial: 2s, factor: 2x).

### Inference Batching
`predict_sentiment` implements a batching strategy (default $n=16$) for FinBERT inference.
- **Scaling Benefit**: Batching significantly reduces the overhead of constant tensor transfers between CPU and GPU/VRAM during large-scale dataset analysis.

### UI State Management & Caching
Streamlit's `@st.cache_resource` and `@st.cache_data` are used strategically:
- **Model Caching**: The 400MB+ FinBERT model is cached in memory to avoid reload latency on UI interaction.
- **Session State**: `st.session_state` is used to persist analysis results, preventing the pipeline from re-running (and incurring API costs) when the user toggles UI filters.

## 4. Architecture & Domain Nuance

### Decoupled Logic
The architecture strictly separates UI (Streamlit in `app.py`) from logic (`src/core_functions.py`).
- **The Why**: This allows the core sentiment and OCR logic to be imported into external scripts (like `generate_materials.py`) or notebooks without triggering the Streamlit runtime.

### Domain-Specific Confidence Thresholds
A heuristic threshold of **0.6 (60%)** is applied to FinBERT predictions.
- **The Why**: Financial sentiment is often subtle. Predictions falling below this threshold are flagged as "uncertain" rather than providing a potentially misleading classification.

### Data Sensitivity & sanitation
The `.gitignore` is configured to exclude `streamlit_data/`, preventing large binary assets and generated JSON predictions from being committed. This keeps the repository lightweight while protecting potentially sensitive local data outputs.

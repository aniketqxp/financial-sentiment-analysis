# Financial Sentiment Analysis Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A high-fidelity sentiment analysis pipeline designed for financial text. This system integrates Optical Character Recognition (OCR), Large Language Models (LLMs) for text structuring, and specialized financial transformers (FinBERT) to provide actionable insights with local explainability (LIME).

## System Architecture

```mermaid
graph TD
    %% Node Definitions
    A[/Financial Image or Text/] --> B(OCR Engine: Tesseract)
    B --> C([LLM Correction: Llama 3.1])
    C --> D[FinBERT Classification]
    D --> E{Confidence Check}
    E -->|>=0.6| F([Explainability: LIME])
    E -->|<0.6| G[Flag as Uncertain]
    F --> H[Dashboard Logic]
    G --> H
    H --> I>Sentiment Analytics]
    I --> J[/Visual Insights & Predictions/]

    %% Styling
    style A fill:#2d3436,stroke:#000,color:#fff
    style B fill:#0984e3,stroke:#000,color:#fff
    style C fill:#6c5ce7,stroke:#000,color:#fff
    style D fill:#00b894,stroke:#000,color:#fff
    style E fill:#fdcb6e,stroke:#000,color:#000
    style F fill:#e17055,stroke:#000,color:#fff
    style G fill:#d63031,stroke:#000,color:#fff
    style H fill:#00b894,stroke:#000,color:#fff
    style I fill:#2d3436,stroke:#000,color:#fff
    style J fill:#2d3436,stroke:#000,color:#fff
```

## Key Features
- **OCR-to-Insights**: Extract text from financial reports or screenshots using Pytesseract.
- **LLM-Enhanced Refinement**: Automatic correction of OCR artifacts and sentence segmentation using Llama 3.1.
- **Financial Sentiment Engine**: Domain-specific classification using the ProsusAI/finbert transformer.
- **Model Explainability**: Word-level contribution analysis via LIME to demystify "black-box" predictions.

![Model Output Showcase](./assets/predictions.png)
*Figure 1: Sentiment analysis results showing confidence scores and LIME-based word contributions for each classification.*


## Getting Started

### Prerequisites
- Python 3.9+
- Tesseract OCR Engine installed on your system

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aniketqxp/financial-sentiment-analysis.git
   cd financial-sentiment-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Configure environment variables:
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACE_API_TOKEN=your_token_here
   GEMINI_API_KEY=your_key_here
   ```

## Quick Start

To launch the interactive dashboard:
```bash
streamlit run app.py
```

### Pre-generating Analysis Materials
For large datasets, use the utility script to pre-compute visualizations:
```bash
python scripts/generate_materials.py
```

## Visualizations

![Sentiment Distribution](./assets/sentiment_distribution.png)
![Confidence Profile](./assets/confidence_distribution.png)
![Performance Matrix](./assets/confusion_matrix.png)

## License
Distributed under the MIT License. See `LICENSE` for more information.

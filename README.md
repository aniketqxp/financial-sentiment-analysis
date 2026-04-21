# Financial Sentiment Analysis Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A high-fidelity sentiment analysis pipeline designed for financial text. This system integrates Optical Character Recognition (OCR), Large Language Models (LLMs) for text structuring, and specialized financial transformers (FinBERT) to provide actionable insights with local explainability (LIME).

## System Architecture

```mermaid
graph TD
    %% Node Definitions
    Input[/Financial Image or Text/]
    OCR[OCR Engine: Tesseract]
    LLM([LLM Correction: Llama 3.1])
    FinBERT[[Sentiment Engine: FinBERT]]
    LIME{{Explainability: LIME}}
    Dashboard[Interactive Streamlit Dashboard]
    Insights[/Visual Insights/]

    %% Connections
    Input --> OCR
    OCR --> LLM
    LLM --> FinBERT
    FinBERT --> LIME
    LIME --> Dashboard
    Dashboard --> Insights

    %% Styling
    style Input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style OCR fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style LLM fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style FinBERT fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style LIME fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Dashboard fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style Insights fill:#e1f5fe,stroke:#01579b,stroke-width:2px
```

## Key Features
- **OCR-to-Insights**: Extract text from financial reports or screenshots using Pytesseract.
- **LLM-Enhanced Refinement**: Automatic correction of OCR artifacts and sentence segmentation using Llama 3.1.
- **Financial Sentiment Engine**: Domain-specific classification using the ProsusAI/finbert transformer.
- **Model Explainability**: Word-level contribution analysis via LIME to demystify "black-box" predictions.

<!-- ![Dashboard Overview Placeholder] -->

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

<!-- ![Sentiment Distribution Placeholder] -->
<!-- ![LIME Explanation Placeholder] -->

## License
Distributed under the MIT License. See `LICENSE` for more information.

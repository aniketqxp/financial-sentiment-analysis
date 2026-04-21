"""
Financial Sentiment Analysis Dashboard
A comprehensive Streamlit app for analyzing financial text sentiment using:
OCR -> LLM Cleaning -> FinBERT Classification -> LIME Explainability
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import os
from pathlib import Path
import time
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Import from src/core_functions.py
from src.core_functions import (
    load_finbert_model,
    process_user_input,
    predict_sentiment,
    explain_single_prediction,
    logger
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Card containers */
    .card {
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        background: white;
        border-left: 4px solid #3498db;
    }
    
    .card-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    /* Progress indicators */
    .step-indicator {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    h2 {
        color: #34495e;
        font-weight: 600;
        margin-top: 30px;
    }
    
    h3 {
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: bold;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9em;
    }
    
    /* Sample text styling */
    .sample-box {
        background: #fff3cd;
        border: 2px dashed #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_precomputed_data():
    """Load all pre-computed analysis materials"""
    data_dir = Path('./streamlit_data')
    
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        st.info("Please run `python generate_materials.py` first to generate the dataset analysis.")
        return None, None, None, None
    
    try:
        # Load predictions
        predictions_df = pd.read_csv(data_dir / 'dataset_predictions.csv')
        
        # Load explanations
        with open(data_dir / 'interesting_explanations.json', 'r') as f:
            explanations = json.load(f)
        
        # Load analysis summary
        with open(data_dir / 'analysis_summary.json', 'r') as f:
            analysis = json.load(f)
        
        
        return predictions_df, explanations, analysis
        
    except Exception as e:
        st.error(f"Error loading pre-computed data: {str(e)}")
        return None, None, None, None


def render_sentiment_badge(sentiment):
    """Render a colored sentiment badge"""
    sentiment_lower = sentiment.lower() 
    badge_class = f"badge-{sentiment_lower}"
    return f'<span class="badge {badge_class}">{sentiment.upper()}</span>'


def render_lime_explanation(sentence, word_contributions, predicted_label):
    """
    Convert sentence + word weights into HTML with highlighted words
    Uses size, boldness, and color to show contribution strength
    
    Args:
        sentence: str - full sentence
        word_contributions: list of [word, weight] pairs from LIME
        predicted_label: str - 'positive', 'negative', or 'neutral'
    
    Note: LIME weights are counterintuitive - negative weights mean removing 
    the word HURTS the prediction (i.e., the word SUPPORTS it)
    """
    word_weights = {word.lower(): weight for word, weight in word_contributions}
    words = sentence.split()
    
    html_parts = []
    for word in words:
        clean_word = ''.join(c for c in word if c.isalnum()).lower()
        
        if clean_word in word_weights:
            weight = word_weights[clean_word]
            abs_weight = abs(weight)
            
            # Flip color interpretation based on predicted label
            # Negative LIME weight = word SUPPORTS the prediction
            if predicted_label == 'positive':
                # Negative weight = supports positive = green
                # Positive weight = contradicts positive = red
                color = "46, 204, 113" if weight < 0 else "231, 76, 60"
            elif predicted_label == 'negative':
                # Negative weight = supports negative = red
                # Positive weight = contradicts negative = green
                color = "231, 76, 60" if weight < 0 else "46, 204, 113"
            else:  # neutral
                # For neutral, use gray with varying intensity
                color = "149, 165, 166"
            
            # Scale opacity based on weight (0.2 to 0.9)
            opacity = min(0.2 + abs_weight * 4, 0.9)
            
            # Scale font size based on weight (1em to 1.4em)
            font_size = 1 + min(abs_weight * 2, 0.4)
            
            # Bold if significant
            font_weight = "700" if abs_weight > 0.15 else "500"
            
            # Add border for very strong contributions
            border = f"2px solid rgba({color}, {opacity})" if abs_weight > 0.15 else "none"
            
            html_parts.append(
                f'<span class="lime-word" style="'
                f'background-color: rgba({color}, {opacity}); '
                f'font-weight: {font_weight}; '
                f'font-size: {font_size}em; '
                f'border: {border}; '
                f'padding: 2px 4px; '
                f'border-radius: 3px; '
                f'margin: 0 2px; '
                f'" title="Weight: {weight:+.3f} (supports {predicted_label})">{word}</span>'
            )
        else:
            html_parts.append(word)
    
    return ' '.join(html_parts)

def render_explanation_card(explanation):
    """Render an explanation card with LIME highlighting"""
    sentiment = explanation['predicted_label']
    confidence = explanation['confidence'] * 100
    
    # Generate word HTML
    lime_html = render_lime_explanation(
        explanation['sentence'], 
        explanation['word_contributions'],
        explanation['predicted_label']
    )

# Wherever else you call it for live demo

    
    # Sentiment colors and labels
    sentiment_config = {
        'positive': {'color': '#10b981', 'bg': '#d1fae5', 'label': 'POSITIVE'},
        'neutral': {'color': '#6b7280', 'bg': '#f3f4f6', 'label': 'NEUTRAL'},
        'negative': {'color': '#ef4444', 'bg': '#fee2e2', 'label': 'NEGATIVE'},
    }
    
    config = sentiment_config.get(sentiment.lower(), sentiment_config['neutral'])
    
    # Build complete HTML with inline styles
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                padding: 0;
                background: transparent;
            }}
            
            .explanation-container {{
                background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                border: 2px solid {config['color']}40;
                position: relative;
                overflow: hidden;
            }}
            
            .explanation-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 5px;
                background: linear-gradient(90deg, {config['color']}, {config['color']}80);
            }}
            
            .explanation-header {{
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 24px;
                padding-bottom: 16px;
                border-bottom: 2px solid #e5e7eb;
                flex-wrap: wrap;
            }}
            
            .sentiment-badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 20px;
                border-radius: 12px;
                background: {config['bg']};
                color: {config['color']};
                font-weight: 700;
                font-size: 1.1em;
                border: 2px solid {config['color']};
                box-shadow: 0 4px 12px {config['color']}30;
            }}
            
            .confidence-meter {{
                display: flex;
                align-items: center;
                gap: 12px;
                flex: 1;
                min-width: 200px;
            }}
            
            .confidence-bar-container {{
                flex: 1;
                height: 24px;
                background: #e5e7eb;
                border-radius: 12px;
                overflow: hidden;
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .confidence-bar {{
                height: 100%;
                background: linear-gradient(90deg, {config['color']}, {config['color']}cc);
                width: {confidence}%;
                transition: width 0.6s ease-out;
                position: relative;
                box-shadow: 0 0 10px {config['color']}60;
            }}
            
            .confidence-bar::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            .confidence-text {{
                font-weight: 700;
                font-size: 1.2em;
                color: {config['color']};
                min-width: 60px;
                text-align: right;
            }}
            
            .reason-tag {{
                color: #6b7280;
                font-style: italic;
                font-size: 0.95em;
                background: white;
                padding: 6px 12px;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }}
            
            .explanation-sentence {{
                background: white;
                padding: 24px;
                border-radius: 12px;
                margin: 20px 0;
                font-size: 1.15em;
                line-height: 2;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e5e7eb;
            }}
            
            .lime-word {{
                padding: 4px 8px;
                margin: 0 3px;
                border-radius: 6px;
                transition: all 0.2s ease;
                cursor: help;
                display: inline-block;
                position: relative;
            }}
            
            .lime-word:hover {{
                transform: scale(1.15) translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                z-index: 10;
            }}
            
            .lime-word::after {{
                content: attr(title);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%) translateY(-8px);
                background: #1f2937;
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 0.85em;
                white-space: nowrap;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s, transform 0.2s;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 1000;
            }}
            
            .lime-word:hover::after {{
                opacity: 1;
                transform: translateX(-50%) translateY(-4px);
            }}
            
            .legend {{
                display: flex;
                gap: 24px;
                margin-top: 20px;
                padding: 16px;
                background: white;
                border-radius: 10px;
                font-size: 0.95em;
                border: 1px solid #e5e7eb;
                flex-wrap: wrap;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .legend-color {{
                width: 24px;
                height: 24px;
                border-radius: 6px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }}
            
            .legend-label {{
                font-weight: 500;
                color: #374151;
            }}
        </style>
    </head>
    <body>
        <div class="explanation-container">
            <div class="explanation-header">
                <div class="sentiment-badge">
                    <span>{config['label']}</span>
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-bar-container">
                        <div class="confidence-bar"></div>
                    </div>
                    <div class="confidence-text">{confidence:.1f}%</div>
                </div>
                
                <div class="reason-tag">
                    {explanation['reason']}
                </div>
            </div>
            
            <div class="explanation-sentence">
                {lime_html}
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.7), rgba(46, 204, 113, 0.9));"></div>
                    <span class="legend-label">Positive Influence</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.7), rgba(231, 76, 60, 0.9));"></div>
                    <span class="legend-label">Negative Influence</span>
                </div>
                <div class="legend-item">
                    <span class="legend-label"><strong>Bold & Larger</strong> = Strong Impact (|weight| > 0.15)</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(html, height=350, scrolling=False)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Logo and branding
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("assets/logo.png", width=60)
    with col2:
        st.markdown("### Financial AI")
        st.caption("Sentiment Analysis")
    
    st.markdown("---")
    
    # Navigation section with custom styling
    st.markdown("### Navigation")
    section = st.radio(
        "Select Section:",
        ["Overview", "Dataset Analysis", "Live Demo"],
        label_visibility="collapsed",
        key="navigation"
    )
    
    st.markdown("---")
    
    # Model Stack
    st.markdown("**Model Stack**")
    st.markdown("""
    <div style="font-size: 0.9em; line-height: 2; padding-left: 8px;">
        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
            <span style="color: #888;">OCR</span>
            <strong>Tesseract</strong>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
            <span style="color: #888;">LLM</span>
            <strong>Llama 3.1 8B</strong>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
            <span style="color: #888;">Classifier</span>
            <strong>FinBERT</strong>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
            <span style="color: #888;">XAI</span>
            <strong>LIME</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    # Live Demo Settings
    if section == "Live Demo":
        st.markdown("---")
        st.markdown("### Settings")
        
        # Settings card with subtle background
        st.markdown("""
        <div style="
            background-color: rgba(99, 110, 250, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        ">
        """, unsafe_allow_html=True)
        
        batch_size = st.slider(
            "Batch Size",
            min_value=8,
            max_value=32,
            value=16,
            help="Number of samples to process at once"
        )
        
        show_intermediate = st.checkbox(
            "Show Intermediate Steps",
            value=True,
            help="Display OCR and LLM outputs"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px; color: #666;">
        <p style="font-size: 0.8em; margin: 5px 0;">
            Financial Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# ============================================================================
# OVERVIEW SECTION
# ============================================================================

if section == "Overview":
    # Dataset Information
    st.markdown("### About the Dataset: Auditor Sentiment")

    # Dataset stats from HF card
    total_sentences = "4,840" 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Sentences",
            value=total_sentences,
            help="Sentences from English financial news"
        )

    with col2:
        st.metric(
            label="Categories",
            value="3 (P/N/Neu)",
            help="'positive', 'negative', 'neutral' labels"
        )

    with col3:
        st.metric(
            label="Annotation",
            value="16 Auditors",
            help="Annotated by auditors with >75% agreement"
        )

    with col4:
        st.metric(
            label="Source",
            value="Financial News",
            help="English news reports from European markets"
        )

    st.markdown("""
    **Auditor Sentiment Dataset** contains **4,840 sentences** extracted from English-language financial news 
    reports. Each sentence was annotated by **16 auditors** with finance expertise, achieving **>75% inter-annotator 
    agreement**. Labels reflect sentiment from an **investor's perspective** - how the sentence might influence 
    stock price perception.

    ### Annotation Focus
    Auditors classified sentences based on their **potential impact on the subject company's stock price**. The 
    dataset covers corporate earnings, mergers, acquisitions, market developments, and financial performance reports.

    ### Sentiment Distribution
    The dataset shows typical financial news patterns with **neutral statements dominating**, followed by positive, 
    then negative sentiment (exact distribution available in analysis section).

    **Key Features:**
    - **Domain-specific**: Created specifically for financial sentiment analysis
    - **Expert annotations**: Labeled by finance professionals, not general crowdworkers
    - **Investor perspective**: Sentiment reflects stock market impact, not general opinion
    - **High agreement**: Only sentences with strong consensus included (>75%)
    """)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### Quick Stats")
    
    _, _, analysis = load_precomputed_data()
    
    if analysis:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{analysis['total_samples']:,}")
        
        with col2:
            st.metric("Avg Confidence", f"{analysis['avg_confidence']*100:.1f}%")
        
        with col3:
            if 'accuracy' in analysis:
                st.metric("Accuracy", f"{analysis['accuracy']*100:.1f}%")
            else:
                st.metric("High Confidence", analysis['high_confidence_samples'])
        
        with col4:
            most_common = max(analysis['sentiment_distribution'].items(), key=lambda x: x[1])[0]
            st.metric("Most Common", most_common.title())
    
    st.markdown("---")
    
    # Workflow
    st.markdown("### Pipeline Workflow")

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #667eea; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;"></span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">Image/Text Input</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">User uploads document or pastes text</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; color: #bdc3c7; font-size: 1.3em; margin: -5px 0;">|</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #e74c3c; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;">OCR</span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">OCR Extraction</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">Tesseract extracts raw text from images</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; color: #bdc3c7; font-size: 1.3em; margin: -5px 0;">|</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;"></span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">LLM Structuring</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">Llama 3.1 cleans and structures sentences</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; color: #bdc3c7; font-size: 1.3em; margin: -5px 0;">|</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #2ecc71; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;"></span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">Sentiment Classification</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">FinBERT predicts positive/negative/neutral</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; color: #bdc3c7; font-size: 1.3em; margin: -5px 0;">|</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #f39c12; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;"></span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">LIME Explanation</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">Word-level contribution analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; color: #bdc3c7; font-size: 1.3em; margin: -5px 0;">↓</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #9b59b6; padding: 15px 20px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;"></span>
            <div>
                <div style="font-weight: 600; color: #2c3e50; font-size: 1.05em;">Results & Visualizations</div>
                <div style="font-size: 0.88em; color: #7f8c8d;">Interactive dashboards and insights</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - Multi-Modal Input: Upload images or paste text
        - Pre-trained Models: FinBERT fine-tuned on financial texts
        - Explainable AI: LIME highlights prediction drivers
        """)
    
    with col2:
        st.markdown("""
        - Dataset Analysis: 3,877 pre-analyzed financial sentences
        - Real-time Processing: Live demo with full pipeline
        """)
    
    st.markdown("---")
    
    # Technology & References
    st.markdown("### Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Processing Pipeline:**
        - `Tesseract` - OCR engine
        - `Llama 3.1 8B` - LLM structuring
        - `FinBERT` - Sentiment classifier
        - `LIME` - Model explainability
        """)
    
    with col2:
        st.markdown("""
        **Infrastructure:**
        - `Streamlit` - Web framework
        - `Hugging Face` - Model hosting
        - `Plotly` - Visualizations
        """)
    
    st.markdown("---")
    
    # References
    with st.expander("Key References & Dataset Info"):
        st.markdown("""
        **Academic Papers:**
        - **FinBERT**: [Araci, D. (2019). Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
        - **LIME**: [Ribeiro et al. (2016). "Why Should I Trust You?": Explaining Predictions](https://arxiv.org/abs/1602.04938)
        - **BERT**: [Devlin et al. (2018). Pre-training Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
        
        **Use Cases:**
        - Investment research and report analysis
        - Risk assessment in financial communications
        - Market sentiment tracking over time
        - Automated screening of financial documents
        """)
    
    st.success("Use the sidebar to explore Dataset Analysis or try the Live Demo!")

# ============================================================================
# DATASET ANALYSIS SECTION
# ============================================================================

elif section == "Dataset Analysis":
    st.title("Dataset Analysis")
    st.markdown("Analysis of Financial PhraseBank dataset with 3,877 financial sentences")
    
    # Load data
    predictions_df, explanations, analysis = load_precomputed_data()
    
    if predictions_df is None:
        st.stop()
    
    # --- Dataset Overview ---
    st.title("Financial Sentiment Intelligence")
    st.markdown("### Decision-Support System for Professional Investors")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{len(predictions_df):,}")
    with col2:
        avg_length = predictions_df['sentence'].str.len().mean()
        st.metric("Avg Sentence Length", f"{avg_length:.0f} chars")
    with col3:
        st.metric("Dataset", "Financial PhraseBank")
    
    # Show data toggle
    show_all = st.checkbox("Show all rows", value=False)
    display_df = predictions_df if show_all else predictions_df.head(100)
    
    # Format dataframe for display
    display_cols = ['sentence', 'predicted_label', 'confidence']
    if 'label_name' in display_df.columns:
        display_cols.insert(1, 'label_name')

    # Reset index to start from 1
    display_df_indexed = display_df[display_cols].copy()
    display_df_indexed.index = range(1, len(display_df_indexed) + 1)

    st.dataframe(
        display_df_indexed,
        use_container_width=True,
        height=400,
        column_config={
            "sentence": st.column_config.TextColumn("Sentence", width="large"),
            "predicted_label": st.column_config.TextColumn("Predicted"),
            "label_name": st.column_config.TextColumn("True Label"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        }
    )
    
    if not show_all:
        st.caption(f"Showing 100 of {len(predictions_df)} rows")
    
    # --- Statistical Summary ---
    st.markdown("### Statistical Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{analysis['total_samples']:,}")
    
    with col2:
        st.metric("Average Confidence", f"{analysis['avg_confidence']*100:.1f}%")
    
    with col3:
        if 'accuracy' in analysis:
            st.metric("Model Accuracy", f"{analysis['accuracy']*100:.1f}%")
        else:
            st.metric("High Confidence (>90%)", analysis['high_confidence_samples'])
    
    with col4:
        st.metric("Low Confidence (<60%)", analysis['low_confidence_samples'])
    
    # Most uncertain predictions
    with st.expander("Most Uncertain Predictions", expanded=False):
        uncertain_indices = analysis['most_uncertain'][:5]
        for idx in uncertain_indices:
            row = predictions_df.iloc[idx]
            st.markdown(f"""
            **Index {idx}** - {render_sentiment_badge(row['predicted_label'])} 
            ({row['confidence']*100:.1f}% confidence)
            
            _{row['sentence'][:200]}{'...' if len(row['sentence']) > 200 else ''}_
            """, unsafe_allow_html=True)
            st.markdown("---")
    
    # --- Interactive Visualizations ---
    st.markdown("### Interactive Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Distribution", "Confidence Distribution", "Confusion Matrix"])
    
    with tab1:
        viz_path = Path('./streamlit_data/sentiment_distribution.html')
        if viz_path.exists():
            with open(viz_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=False)
        else:
            st.warning("Visualization not found")
    
    with tab2:
        viz_path = Path('./streamlit_data/confidence_distribution.html')
        if viz_path.exists():
            with open(viz_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=False)
        else:
            st.warning("Visualization not found")
    
    with tab3:
        viz_path = Path('./streamlit_data/confusion_matrix.html')
        if viz_path.exists():
            with open(viz_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=False)
        else:
            st.info("Confusion matrix not available (requires true labels)")
    
    # --- Interesting Samples & LIME Explanations ---
    st.markdown("### Interesting Samples & LIME Explanations")
    st.markdown("These samples showcase how LIME highlights influential words for different predictions")
    
    if explanations:
        for i, exp in enumerate(explanations):
            with st.container():
                render_explanation_card(exp)
                
                # Show top contributing words as bar chart
                with st.expander(f"View Top Contributing Words (Sample {i+1})", expanded=False):
                    words, weights = zip(*exp['word_contributions'][:10])
                    contrib_df = pd.DataFrame({
                        'Word': words,
                        'Contribution': weights
                    })
                    st.bar_chart(contrib_df.set_index('Word'))
    else:
        st.info("No interesting samples available")
    
    # --- Download Section ---
    st.markdown("### Download Data")
    csv_data = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=csv_data,
        file_name="financial_sentiment_predictions.csv",
        mime="text/csv"
    )

# ============================================================================
# LIVE DEMO SECTION
# ============================================================================

elif section == "Live Demo":
    st.title("Live Sentiment Analysis Demo")
    st.markdown("Process your own financial text or images through the complete pipeline")
    
    # --- API Key Input ---
    st.markdown("### API Configuration")

    # Try to get from environment first
    api_token = os.getenv('HUGGINGFACE_API_TOKEN')

    if api_token:
        st.session_state.api_token = api_token
        st.success("API token loaded from environment")
    else:
        api_token = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Required for LLM processing. Get yours at https://huggingface.co/settings/tokens",
            key="hf_token"
        )
        
        if api_token:
            if 'api_token' not in st.session_state or st.session_state.api_token != api_token:
                st.session_state.api_token = api_token
            st.success("API token configured")
        else:
            st.warning("Please provide your Hugging Face API token to continue")
    
    st.markdown("---")
    
    # --- Input Method Selection ---
    st.markdown("### Input Method")
    
    input_method = st.radio(
        "Choose input type:",
        ["Text Input", "Image Upload"],
        horizontal=True
    )
    
    uploaded_file = None
    text_input = None
    
    if input_method == "Image Upload":
        uploaded_file = st.file_uploader(
            "Upload financial document image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload an image containing financial text"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.info(f"""
                **File Details**
                - Name: {uploaded_file.name}
                - Size: {uploaded_file.size / 1024:.1f} KB
                - Format: {image.format}
                """)
    
    else:  # Text Input
        # Initialize sample text in session state if not present
        if 'sample_text' not in st.session_state:
            st.session_state.sample_text = ""
        
        # Sample text button
        if st.button("Use Sample Text"):
            sample_texts = [
                "The company reported strong earnings growth of 25% year-over-year, driven by increased demand.",
                "Operating margins declined due to rising input costs and supply chain disruptions.",
                "The firm maintained its market position with stable revenue throughout the quarter.",
            ]
            st.session_state.sample_text = sample_texts[0]
        
        text_input = st.text_area(
            "Paste financial text",
            value=st.session_state.sample_text,
            height=150,
            placeholder="Example: The company's revenue increased by 15% this quarter, exceeding analyst expectations.",
            help="Enter financial text for sentiment analysis"
        )
        
        # Update session state with current input
        st.session_state.sample_text = text_input
    
    st.markdown("---")
    
    # --- Process Button ---
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        
        # Validation
        if not api_token:
            st.error("Please provide your Hugging Face API token")
            st.stop()
        
        if not uploaded_file and not text_input:
            st.error("Please provide input (upload image or enter text)")
            st.stop()
        
        try:
            # --- Load Model (Once) ---
            if 'model_loaded' not in st.session_state:
                with st.spinner("Loading FinBERT model..."):
                    tokenizer, model = load_finbert_model()
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                tokenizer = st.session_state.tokenizer
                model = st.session_state.model
            
            # --- Step 1: OCR/LLM Processing ---
            st.markdown('<div class="step-indicator"><b>Step 1/3:</b> Processing input...</div>', unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting and structuring text...")
            
            # Save uploaded file temporarily if needed
            image_path = None
            if uploaded_file:
                temp_dir = Path('./temp')
                temp_dir.mkdir(exist_ok=True)
                image_path = temp_dir / uploaded_file.name
                with open(image_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process input
            df, intermediate_results = process_user_input(
                image_path=str(image_path) if image_path else None,
                text_input=text_input,
                api_token=api_token
            )
            
            progress_bar.progress(33)
            status_text.text("Input processed successfully")
            time.sleep(0.5)
            
            # Display intermediate results
            if show_intermediate:
                # OCR Output
                if 'ocr_output' in intermediate_results:
                    with st.expander("OCR Output", expanded=False):
                        ocr_data = intermediate_results['ocr_output']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{ocr_data['metadata']['avg_confidence']:.1f}%")
                        with col2:
                            st.metric("Characters", ocr_data['metadata']['character_count'])
                        with col3:
                            st.metric("Low Confidence Words", ocr_data['metadata']['low_confidence_words'])
                        
                        st.code(ocr_data['text'], language=None)
                
                # LLM Output
                if 'llm_output' in intermediate_results:
                    with st.expander("LLM Structured Output", expanded=True):
                        llm_data = intermediate_results['llm_output']
                        
                        st.info(f"**Transformation**: Extracted **{llm_data['sentence_count']} sentences** from input text")
                        
                        st.markdown("**Structured Sentences:**")
                        for i, sentence in enumerate(llm_data['parsed']['sentences'], 1):
                            st.markdown(f"{i}. {sentence}")
            
            # --- Step 2: Sentiment Prediction ---
            st.markdown('<div class="step-indicator"><b>Step 2/3:</b> Running FinBERT classification...</div>', unsafe_allow_html=True)
            
            status_text.text("Analyzing sentiment...")
            
            predictions = predict_sentiment(
                df['sentence'].tolist(),
                tokenizer,
                model,
                batch_size=batch_size,
                show_progress=False
            )
            
            df = pd.concat([df.reset_index(drop=True), predictions], axis=1)
            
            progress_bar.progress(66)
            status_text.text("Sentiment analysis complete")
            time.sleep(0.5)
            
            # Display prediction results
            st.markdown("### Prediction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sentences", len(df))
            with col2:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
            with col3:
                most_common = df['predicted_label'].mode()[0]
                st.metric("Dominant Sentiment", most_common.title())
            
            # Results table - reindex to start from 1
            df_display = df[['sentence', 'predicted_label', 'confidence']].copy()
            df_display.index = range(1, len(df_display) + 1)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                column_config={
                    "sentence": st.column_config.TextColumn("Sentence", width="large"),
                    "predicted_label": st.column_config.TextColumn("Sentiment"),
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
            
            # --- Step 3: LIME Explanation ---
            st.markdown('<div class="step-indicator"><b>Step 3/3:</b> Generating LIME explanation...</div>', unsafe_allow_html=True)
            
            status_text.text("Creating interpretability visualization...")
            
            # Pick the most confident prediction
            top_idx = df['confidence'].idxmax()
            sentence = df.loc[top_idx, 'sentence']
            label = df.loc[top_idx, 'predicted_label']
            confidence = df.loc[top_idx, 'confidence']
            
            explanation = explain_single_prediction(sentence, label, tokenizer, model)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display LIME explanation
            st.markdown("### LIME Explanation")
            st.markdown(f"Explaining the most confident prediction ({confidence*100:.1f}% confidence)")
            
            explanation_display = {
                'sentence': sentence,
                'predicted_label': label,
                'confidence': confidence,
                'reason': 'Most confident prediction',
                'word_contributions': explanation['word_contributions']
            }
            
            render_explanation_card(explanation_display)
            
            # Show all predictions if multiple sentences
            if len(df) > 1:
                st.markdown("### All Predictions")
                for idx, row in df.iterrows():
                    with st.expander(f"Sentence {idx+1}: {row['predicted_label'].upper()} ({row['confidence']*100:.1f}%)", expanded=False):
                        st.markdown(f"**{row['sentence']}**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive", f"{row['prob_positive']*100:.1f}%")
                        with col2:
                            st.metric("Negative", f"{row['prob_negative']*100:.1f}%")
                        with col3:
                            st.metric("Neutral", f"{row['prob_neutral']*100:.1f}%")
            
            # Success message
            st.success("Analysis complete! See results above.")
            
            # Clean up temp file
            if image_path and image_path.exists():
                image_path.unlink()
        
        except FileNotFoundError as e:
            st.error(f"File not found: {str(e)}")
            st.info("Please make sure the uploaded file is valid")
        
        except ValueError as e:
            st.error(f"Invalid input: {str(e)}")
            if "API token" in str(e):
                st.info("Please check your Hugging Face API token")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            with st.expander("View Error Details"):
                st.exception(e)
            st.info("Please try again or contact support if the issue persists")
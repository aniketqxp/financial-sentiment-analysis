import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core_functions import load_finbert_model, predict_sentiment, retry_with_backoff, logger
from datasets import load_dataset

@retry_with_backoff(max_retries=3, initial_wait=2)
def load_financial_data():
    """Load Financial PhraseBank dataset"""
    logger.info("Loading Financial PhraseBank dataset...")
    try:
        dataset = load_dataset('FinanceInc/auditor_sentiment', split='train')
        df = pd.DataFrame(dataset)
        if df['label'].dtype == 'object':
            df['label_name'] = df['label']
        else:
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df['label_name'] = df['label'].map(label_map)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def generate_static_visuals(output_dir='./assets'):
    """Generate static PNG visualizations for README and documentation"""
    
    print(f"Generating static visuals in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data and Model
    tokenizer, model = load_finbert_model()
    df = load_financial_data()
    
    # 2. Run Inference on a subset for speed/demo
    print("Running inference on dataset subset...")
    subset_df = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
    predictions = predict_sentiment(subset_df['sentence'].tolist(), tokenizer, model)
    subset_df = pd.concat([subset_df, predictions], axis=1)
    
    # 3. Sentiment Distribution Plot
    print("Creating sentiment distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    counts = subset_df['predicted_label'].value_counts()
    colors = ['#2ecc71', '#95a5a6', '#e74c3c'] # Green, Gray, Red
    
    sns.barplot(x=counts.index, y=counts.values, palette=colors, hue=counts.index, legend=False)
    plt.title('Financial Sentiment Distribution', fontsize=15, fontweight='bold')
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Sentences', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    
    # 4. Confidence Distribution Plot
    print("Creating confidence distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(subset_df['confidence'], bins=20, kde=True, color='#3498db')
    plt.title('Prediction Confidence Profile', fontsize=15, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    
    # 5. Confusion Matrix (if ground truth exists)
    if 'label_name' in subset_df.columns:
        print("Creating confusion matrix...")
        plt.figure(figsize=(8, 6))
        cm = pd.crosstab(subset_df['label_name'], subset_df['predicted_label'], 
                         rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Model Performance: Confusion Matrix', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)

    print(f"\nVisuals generated successfully in {output_dir}")

if __name__ == "__main__":
    generate_static_visuals()

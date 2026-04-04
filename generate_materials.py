# generate_materials.py - Pre-generate all dataset analysis materials for Streamlit

from core_functions import *
from datasets import load_dataset
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET LOADING
# ============================================================================

@retry_with_backoff(max_retries=3, initial_wait=2)
def load_financial_data():
    """Load Financial PhraseBank dataset"""
    logger.info("Loading Financial PhraseBank dataset...")
    
    try:
        dataset = load_dataset('FinanceInc/auditor_sentiment', split='train')
        
        if len(dataset) == 0:
            raise ValueError("Downloaded dataset is empty")
        
        df = pd.DataFrame(dataset)
        
        if 'sentence' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset missing required columns")
        
        logger.info(f"Loaded {len(df)} sentences")
        
        if df['label'].dtype == 'object':
            df['label_name'] = df['label']
        else:
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df['label_name'] = df['label'].map(label_map)
        
        logger.info(f"Label distribution: {df['label_name'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def identify_interesting_samples(df, sentences, top_n=4):
    """Identify most interesting samples for explanation visualization"""
    logger.info("Identifying interesting samples")
    
    interesting_indices = []
    
    # Most confident per class
    for label in df['predicted_label'].unique():
        label_df = df[df['predicted_label'] == label]
        if not label_df.empty:
            top_idx = label_df.nlargest(1, 'confidence').index[0]
            interesting_indices.append({
                'index': int(top_idx),
                'reason': f'Most confident {label}',
                'confidence': float(df.loc[top_idx, 'confidence'])
            })
    
    # Most uncertain
    uncertain = df.nsmallest(2, 'confidence')
    for idx in uncertain.index:
        interesting_indices.append({
            'index': int(idx),
            'reason': 'Uncertain prediction',
            'confidence': float(df.loc[idx, 'confidence'])
        })
    
    # Misclassifications
    if 'label_name' in df.columns:
        misclassified = df[df['predicted_label'] != df['label_name']]
        if not misclassified.empty:
            for idx in misclassified.head(2).index:
                interesting_indices.append({
                    'index': int(idx),
                    'reason': 'Misclassified',
                    'confidence': float(df.loc[idx, 'confidence'])
                })
    
    # Remove duplicates
    unique_indices = []
    seen = set()
    for item in interesting_indices:
        if item['index'] not in seen:
            unique_indices.append(item)
            seen.add(item['index'])
    
    return unique_indices[:top_n]


def generate_explanations_for_interesting_samples(df, sentences, interesting_samples, tokenizer, model):
    """Generate LIME explanations for interesting samples"""
    logger.info(f"Generating explanations for {len(interesting_samples)} samples")
    
    explainer, predictor_fn = setup_lime_explainer(tokenizer, model)
    explanations = []
    
    for sample in tqdm(interesting_samples, desc="Generating LIME explanations"):
        idx = sample['index']
        
        try:
            text = sentences[idx]
            predicted_label = df.loc[idx, 'predicted_label']
            
            exp = explainer.explain_instance(text, predictor_fn, num_features=10, num_samples=500)
            
            explanations.append({
                'index': idx,
                'sentence': text,
                'predicted_label': predicted_label,
                'confidence': float(df.loc[idx, 'confidence']),
                'reason': sample['reason'],
                'word_contributions': exp.as_list()
            })
            
        except Exception as e:
            logger.warning(f"Failed to generate explanation for index {idx}: {str(e)}")
            continue
    
    return explanations


def analyze_results(df):
    """Generate summary statistics"""
    logger.info("Analyzing results...")
    
    analysis = {
        'total_samples': len(df),
        'sentiment_distribution': df['predicted_label'].value_counts().to_dict(),
        'avg_confidence': float(df['confidence'].mean()),
        'high_confidence_samples': int(len(df[df['confidence'] > 0.9])),
        'low_confidence_samples': int(len(df[df['confidence'] < 0.6])),
        'most_uncertain': [int(idx) for idx in df.nsmallest(5, 'confidence').index.tolist()]
    }
    
    if 'label_name' in df.columns:
        analysis['accuracy'] = float((df['predicted_label'] == df['label_name']).mean())
        confusion_data = pd.crosstab(df['label_name'], df['predicted_label'])
        analysis['confusion_matrix'] = confusion_data.to_dict()
    
    return analysis


def create_visualizations(df, analysis, output_dir):
    """Create Plotly visualizations"""
    logger.info("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sentiment distribution
    sentiment_counts = df['predicted_label'].value_counts()
    color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
    
    fig1 = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'},
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map=color_map
    )
    fig1.update_layout(showlegend=False, title_font_size=16)
    fig1.write_html(os.path.join(output_dir, 'sentiment_distribution.html'))
    
    # Confidence distribution
    fig2 = px.histogram(df, x='confidence', nbins=30, title='Prediction Confidence Distribution')
    fig2.update_traces(marker_color='#3498db', marker_line_color='black', marker_line_width=1)
    fig2.write_html(os.path.join(output_dir, 'confidence_distribution.html'))
    
    # Confusion matrix
    if 'label_name' in df.columns and 'confusion_matrix' in analysis:
        conf_df = pd.DataFrame(analysis['confusion_matrix'])
        fig3 = go.Figure(data=go.Heatmap(
            z=conf_df.values,
            x=conf_df.columns,
            y=conf_df.index,
            colorscale='Blues',
            text=conf_df.values,
            texttemplate='%{text}',
            textfont={"size": 14}
        ))
        fig3.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')
        fig3.write_html(os.path.join(output_dir, 'confusion_matrix.html'))
    
    logger.info("Visualizations saved")


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_all_materials(output_dir='./streamlit_data'):
    """Generate all pre-computed materials for Streamlit app"""
    
    logger.info("="*60)
    logger.info("GENERATING MATERIALS FOR STREAMLIT APP")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = load_financial_data()
    sentences = df['sentence'].tolist()
    
    # Load model
    tokenizer, model = load_finbert_model()
    
    # Run predictions
    predictions_df = predict_sentiment(sentences, tokenizer, model, batch_size=16)
    df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    
    # Save predictions
    df.to_csv(os.path.join(output_dir, 'dataset_predictions.csv'), index=False)
    logger.info("Saved dataset predictions")
    
    # Identify interesting samples
    interesting_samples = identify_interesting_samples(df, sentences, top_n=4)
    
    # Generate explanations
    explanations = generate_explanations_for_interesting_samples(df, sentences, interesting_samples, tokenizer, model)
    
    # Save explanations
    with open(os.path.join(output_dir, 'interesting_explanations.json'), 'w') as f:
        json.dump(explanations, f, indent=2)
    logger.info("Saved interesting explanations")
    
    # Analysis
    analysis = analyze_results(df)
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info("Saved analysis summary")
    
    # Visualizations
    create_visualizations(df, analysis, output_dir)
    
    logger.info("="*60)
    logger.info("MATERIAL GENERATION COMPLETE")
    logger.info("="*60)
    
    print(f"\nGenerated files in {output_dir}:")
    print("- dataset_predictions.csv")
    print("- interesting_explanations.json")
    print("- analysis_summary.json")
    print("- sentiment_distribution.html")
    print("- confidence_distribution.html")
    print("- confusion_matrix.html")

if __name__ == "__main__":
    print(f"\nOutput directory: {os.path.abspath('./streamlit_data')}")
    print("="*60 + "\n")
    
    generate_all_materials(output_dir='./streamlit_data')
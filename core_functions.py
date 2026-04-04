# core_functions.py - Shared functions for both material generation and Streamlit app

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import json
from lime.lime_text import LimeTextExplainer
import os
from dotenv import load_dotenv
from google import genai
import time
import logging
from functools import wraps
from tqdm import tqdm
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DECORATORS
# ============================================================================

def retry_with_backoff(max_retries=3, initial_wait=2, backoff_factor=2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"{func.__name__} - Attempt {attempt + 1}/{max_retries}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def validate_input(input_type='path'):
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if input_type == 'path' and len(args) > 0:
                path = args[0]
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
            elif input_type == 'api_token' and 'api_token' in kwargs:
                token = kwargs.get('api_token')
                if not token or len(token) < 10:
                    raise ValueError("Invalid API token provided")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_minute=10):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_call = time.time()


# Initialize rate limiters
llama_rate_limiter = RateLimiter(calls_per_minute=10)
gemini_rate_limiter = RateLimiter(calls_per_minute=15)

# ============================================================================
# OCR + LLM PROCESSING
# ============================================================================

@validate_input(input_type='path')
def ocr_image(image_path):
    """Extract text from image with preprocessing and metadata"""
    logger.info(f"Starting OCR on: {image_path}")
    
    try:
        supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext not in supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}. Supported: {supported_formats}")
        
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image)
        
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        metadata = {
            'image_path': image_path,
            'image_size': image.size,
            'format': file_ext,
            'character_count': len(text),
            'avg_confidence': avg_confidence,
            'low_confidence_words': sum(1 for c in confidences if c < 60)
        }
        
        logger.info(f"OCR complete - Confidence: {avg_confidence:.1f}%, Characters: {len(text)}")
        
        return text.strip(), metadata
        
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise


@retry_with_backoff(max_retries=3, initial_wait=2)
@validate_input(input_type='api_token')
def llm_process(text, system_prompt, api_token, timeout=30):
    """Pass text through LLM with validation and error handling"""
    
    if not api_token or len(api_token) < 20:
        raise ValueError("Invalid or missing API token")
    
    llama_rate_limiter.wait()
    
    logger.info(f"Sending request to LLM - Text length: {len(text)}")
    
    url = "https://router.huggingface.co/v1/chat/completions"
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "max_tokens": 2000,
        "temperature": 0.1
    }
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            raise requests.HTTPError(f"API error: {response.status_code}")
        
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            raise ValueError("Invalid API response structure")
        
        content = result["choices"][0]["message"]["content"]
        logger.info(f"LLM response received - Length: {len(content)}")
        
        return content
        
    except requests.Timeout:
        logger.error("LLM request timed out")
        raise
    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise


def process_user_input(image_path=None, text_input=None, api_token=None):
    """Process user input (image or text) through OCR/LLM pipeline"""
    
    intermediate_results = {}
    
    if image_path:
        logger.info("Processing user image")
        ocr_text, ocr_metadata = ocr_image(image_path)
        
        intermediate_results['ocr_output'] = {
            'text': ocr_text,
            'metadata': ocr_metadata
        }
        
        input_text = ocr_text
        
    elif text_input:
        logger.info("Processing user text")
        intermediate_results['raw_text'] = text_input
        input_text = text_input
    else:
        raise ValueError("Must provide either image_path or text_input")
    
    # LLM structuring
    system_prompt = """You are a text processing assistant. Your task is to:
                        1. Clean and correct any OCR errors in the provided text (if present)
                        2. Split the text into individual sentences
                        3. Return a JSON object with this structure: {"sentences": ["sentence1", "sentence2", ...]}

Only return valid JSON, no additional text or explanation."""
    
    llm_output = llm_process(input_text, system_prompt, api_token)
    
    intermediate_results['llm_output'] = {'raw': llm_output}
    
    try:
        sentences_json = json.loads(llm_output)
        
        if 'sentences' not in sentences_json or not isinstance(sentences_json['sentences'], list):
            raise ValueError("LLM output missing 'sentences' array")
        
        df = pd.DataFrame(sentences_json['sentences'], columns=['sentence'])
        intermediate_results['llm_output']['parsed'] = sentences_json
        intermediate_results['llm_output']['sentence_count'] = len(df)
        
        logger.info(f"Successfully extracted {len(df)} sentences")
        
        return df, intermediate_results
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON output: {str(e)}")
        raise ValueError("LLM returned invalid JSON format")


# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

@retry_with_backoff(max_retries=2, initial_wait=3)
def load_finbert_model():
    """Load FinBERT model and tokenizer with error handling"""
    logger.info("Loading FinBERT model...")
    
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        logger.info("Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def predict_sentiment(texts, tokenizer, model, batch_size=16, show_progress=True):
    """Run FinBERT inference on list of texts"""
    logger.info(f"Running inference on {len(texts)} samples")
    
    results = []
    label_names = ['positive', 'negative', 'neutral']
    truncation_warnings = 0
    
    iterator = tqdm(range(0, len(texts), batch_size), desc="Predicting") if show_progress else range(0, len(texts), batch_size)
    
    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        
        for text in batch_texts:
            tokens = tokenizer.tokenize(text)
            if len(tokens) > 510:
                truncation_warnings += 1
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            pred_label_idx = torch.argmax(probs).item()
            pred_label = label_names[pred_label_idx]
            confidence = probs[0][pred_label_idx].item()
            
            results.append({
                'predicted_label': pred_label,
                'confidence': confidence,
                'prob_positive': probs[0][0].item(),
                'prob_negative': probs[0][1].item(),
                'prob_neutral': probs[0][2].item()
            })
    
    if truncation_warnings > 0:
        logger.warning(f"{truncation_warnings} texts were truncated to 512 tokens")
    
    logger.info(f"Inference complete")
    return pd.DataFrame(results)


# ============================================================================
# EXPLAINABILITY WITH LIME
# ============================================================================

def setup_lime_explainer(tokenizer, model):
    """Create LIME explainer for FinBERT"""
    label_names = ['positive', 'negative', 'neutral']
    
    def predictor_fn(texts):
        predictions = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.append(probs[0].numpy())
        return np.array(predictions)
    
    explainer = LimeTextExplainer(class_names=label_names)
    return explainer, predictor_fn


def explain_single_prediction(sentence, predicted_label, tokenizer, model):
    """Generate LIME explanation for a single prediction"""
    logger.info("Generating LIME explanation")
    
    explainer, predictor_fn = setup_lime_explainer(tokenizer, model)
    
    try:
        exp = explainer.explain_instance(
            sentence, 
            predictor_fn, 
            num_features=10,
            num_samples=500
        )
        
        word_weights = exp.as_list()
        
        explanation = {
            'sentence': sentence,
            'predicted_label': predicted_label,
            'word_contributions': word_weights
        }
        
        logger.info("LIME explanation generated")
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {str(e)}")
        raise
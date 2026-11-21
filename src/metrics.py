"""
Metrics Module for UKM Live Notulen ASR
Provides evaluation metrics for ASR model comparison.

Metrics included:
- WER (Word Error Rate)
- CER (Character Error Rate)
- RTF (Real-Time Factor)
- Latency measurement
- Memory usage tracking
- Code-switching accuracy
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from jiwer import wer as jiwer_wer, cer as jiwer_cer


def calculate_wer(references: List[str], predictions: List[str]) -> float:
    """
    Calculate Word Error Rate.
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words
    
    Args:
        references: List of ground truth transcripts
        predictions: List of model predictions
    
    Returns:
        WER as a float (0.0 to 1.0)
    """
    if len(references) == 0 or len(predictions) == 0:
        return 1.0
    
    # Normalize text
    refs = [r.lower().strip() for r in references]
    preds = [p.lower().strip() for p in predictions]
    
    return jiwer_wer(refs, preds)


def calculate_cer(references: List[str], predictions: List[str]) -> float:
    """
    Calculate Character Error Rate.
    
    Same as WER but at character level - useful for agglutinative languages
    and comparing models with different tokenization.
    
    Args:
        references: List of ground truth transcripts
        predictions: List of model predictions
    
    Returns:
        CER as a float (0.0 to 1.0)
    """
    if len(references) == 0 or len(predictions) == 0:
        return 1.0
    
    # Normalize text
    refs = [r.lower().strip() for r in references]
    preds = [p.lower().strip() for p in predictions]
    
    return jiwer_cer(refs, preds)


def calculate_rtf(audio_duration: float, inference_time: float) -> float:
    """
    Calculate Real-Time Factor.
    
    RTF = inference_time / audio_duration
    RTF < 1.0 means faster than real-time (suitable for streaming)
    
    Args:
        audio_duration: Duration of audio in seconds
        inference_time: Time taken for inference in seconds
    
    Returns:
        RTF as a float
    """
    if audio_duration <= 0:
        return float('inf')
    return inference_time / audio_duration


def measure_latency(
    model_fn: callable,
    audio_sample: np.ndarray,
    device: str = 'cuda',
    warmup_runs: int = 3,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure inference latency with warmup.
    
    Args:
        model_fn: Function that takes audio and returns transcript
        audio_sample: Audio array (16kHz)
        device: 'cuda' or 'cpu'
        warmup_runs: Number of warmup iterations
        num_runs: Number of timed iterations
    
    Returns:
        Dictionary with mean, std, min, max latency in seconds
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = model_fn(audio_sample)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = model_fn(audio_sample)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        latencies.append(end - start)
    
    latencies = np.array(latencies)
    
    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'median': float(np.median(latencies))
    }


def measure_memory_usage(model, input_shape: tuple = (1, 16000 * 10), device: str = 'cuda') -> Dict[str, float]:
    """
    Measure peak GPU memory usage during inference.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch, samples)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with memory stats in MB
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {'peak_memory_mb': 0.0, 'allocated_mb': 0.0}
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    return {
        'peak_memory_mb': float(peak_memory),
        'allocated_mb': float(allocated)
    }


def calculate_cs_accuracy(
    references: List[str], 
    predictions: List[str],
    cs_markers: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate code-switching point accuracy.
    
    Measures how well the model handles language switching points
    in code-switching speech.
    
    Args:
        references: List of ground truth transcripts
        predictions: List of model predictions
        cs_markers: List of code-switching marker words (optional)
    
    Returns:
        Dictionary with CS accuracy metrics
    """
    if cs_markers is None:
        # Common Indonesian-English code-switching patterns
        cs_markers = [
            # English words commonly used in Indonesian speech
            'meeting', 'deadline', 'presentation', 'project', 'team',
            'feedback', 'update', 'follow up', 'check', 'review',
            'the', 'is', 'are', 'and', 'but', 'for', 'with', 'to'
        ]
    
    total_cs_points = 0
    correct_cs_points = 0
    total_cs_words = 0
    correct_cs_words = 0
    
    for ref, pred in zip(references, predictions):
        ref_lower = ref.lower()
        pred_lower = pred.lower()
        
        # Count CS marker words
        for marker in cs_markers:
            ref_count = ref_lower.split().count(marker.lower())
            pred_count = pred_lower.split().count(marker.lower())
            
            if ref_count > 0:
                total_cs_words += ref_count
                correct_cs_words += min(ref_count, pred_count)
    
    cs_word_accuracy = correct_cs_words / total_cs_words if total_cs_words > 0 else 0.0
    
    return {
        'cs_word_accuracy': cs_word_accuracy,
        'total_cs_words': total_cs_words,
        'correct_cs_words': correct_cs_words
    }


def calculate_detailed_errors(
    references: List[str], 
    predictions: List[str]
) -> Dict[str, int]:
    """
    Calculate detailed error breakdown (substitutions, deletions, insertions).
    
    Args:
        references: List of ground truth transcripts
        predictions: List of model predictions
    
    Returns:
        Dictionary with error counts
    """
    from jiwer import process_words
    
    total_h, total_s, total_d, total_i = 0, 0, 0, 0
    
    for ref, pred in zip(references, predictions):
        ref_words = ref.lower().strip().split()
        pred_words = pred.lower().strip().split()
        
        try:
            output = process_words(ref_words, pred_words)
            total_h += output.hits
            total_s += output.substitutions
            total_d += output.deletions
            total_i += output.insertions
        except Exception:
            # Fallback to simple alignment
            pass
    
    total_words = total_h + total_s + total_d
    
    return {
        'hits': total_h,
        'substitutions': total_s,
        'deletions': total_d,
        'insertions': total_i,
        'total_words': total_words
    }


def evaluate_model(
    model_fn: callable,
    test_dataset,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model_fn: Function that takes audio array and returns transcript
        test_dataset: Dataset with 'audio' and 'text' columns
        device: 'cuda' or 'cpu'
        max_samples: Maximum samples to evaluate (None for all)
        verbose: Print progress
    
    Returns:
        Dictionary with all evaluation metrics
    """
    references = []
    predictions = []
    latencies = []
    audio_durations = []
    
    n_samples = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    
    if verbose:
        print(f"Evaluating {n_samples} samples...")
    
    for i in range(n_samples):
        sample = test_dataset[i]
        audio = sample['audio']['array']
        text = sample['text']
        duration = len(audio) / sample['audio']['sampling_rate']
        
        # Measure latency
        start = time.perf_counter()
        pred = model_fn(audio)
        latency = time.perf_counter() - start
        
        references.append(text)
        predictions.append(pred)
        latencies.append(latency)
        audio_durations.append(duration)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples}")
    
    # Calculate metrics
    wer = calculate_wer(references, predictions)
    cer = calculate_cer(references, predictions)
    rtf = np.mean([l / d for l, d in zip(latencies, audio_durations) if d > 0])
    cs_metrics = calculate_cs_accuracy(references, predictions)
    error_details = calculate_detailed_errors(references, predictions)
    
    return {
        'wer': wer,
        'cer': cer,
        'rtf': float(rtf),
        'avg_latency': float(np.mean(latencies)),
        'cs_accuracy': cs_metrics['cs_word_accuracy'],
        'error_details': error_details,
        'n_samples': n_samples,
        'total_audio_duration': sum(audio_durations),
        'total_inference_time': sum(latencies)
    }


# Demo function
def demo_metrics():
    """Demo of metrics calculation with synthetic data."""
    print("=" * 50)
    print("Metrics Module Demo")
    print("=" * 50)
    
    # Synthetic data
    refs = [
        "hello world this is a test",
        "meeting jam tiga sore",
        "we need to follow up the project"
    ]
    preds = [
        "hello world this is test",
        "meeting jam tiga lima",
        "we need to follow up project"
    ]
    
    print(f"\nReferences: {refs}")
    print(f"Predictions: {preds}")
    
    print(f"\nWER: {calculate_wer(refs, preds):.2%}")
    print(f"CER: {calculate_cer(refs, preds):.2%}")
    print(f"CS Accuracy: {calculate_cs_accuracy(refs, preds)['cs_word_accuracy']:.2%}")
    print(f"Error Details: {calculate_detailed_errors(refs, preds)}")
    
    # RTF example
    print(f"\nRTF example: {calculate_rtf(audio_duration=10.0, inference_time=2.5):.2f}x")


if __name__ == "__main__":
    demo_metrics()

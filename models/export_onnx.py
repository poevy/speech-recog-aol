"""
ONNX Export Script for ASR Models
=================================

Export fine-tuned models to ONNX format for efficient deployment.

Usage:
    python export_onnx.py --model_path outputs/whisper_tiny/final --model_type whisper
    python export_onnx.py --model_path outputs/wav2vec2/final --model_type wav2vec2
"""

import os
import sys
import argparse
import time
import json

import torch
import numpy as np
from transformers import AutoProcessor


def export_whisper_onnx(model_path, output_dir, quantize=True):
    """Export Whisper model to ONNX."""
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    print(f"Loading Whisper model from {model_path}...")
    
    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path, export=True)
    
    # Save ONNX model
    ort_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Quantize if requested
    if quantize:
        print("Quantizing model (int8)...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Find and quantize encoder and decoder ONNX files
        for fname in os.listdir(output_dir):
            if fname.endswith('.onnx') and 'quantized' not in fname:
                fpath = os.path.join(output_dir, fname)
                quant_path = os.path.join(output_dir, fname.replace('.onnx', '_quantized.onnx'))
                quantize_dynamic(fpath, quant_path, weight_type=QuantType.QLosslessAwareInt8)
                print(f"  Quantized: {fname}")
    
    # Get model sizes
    sizes = {}
    for fname in os.listdir(output_dir):
        if fname.endswith('.onnx'):
            fpath = os.path.join(output_dir, fname)
            sizes[fname] = os.path.getsize(fpath) / 1024 / 1024  # MB
    
    return sizes


def export_wav2vec2_onnx(model_path, output_dir, quantize=True):
    """Export Wav2Vec2 model to ONNX."""
    from optimum.onnxruntime import ORTModelForCTC
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    
    print(f"Loading Wav2Vec2 model from {model_path}...")
    
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export
    print("Exporting to ONNX...")
    ort_model = ORTModelForCTC.from_pretrained(model_path, export=True)
    ort_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Quantize
    if quantize:
        print("Quantizing model (int8)...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        for fname in os.listdir(output_dir):
            if fname.endswith('.onnx') and 'quantized' not in fname:
                fpath = os.path.join(output_dir, fname)
                quant_path = os.path.join(output_dir, fname.replace('.onnx', '_quantized.onnx'))
                quantize_dynamic(fpath, quant_path, weight_type=QuantType.QLosslessAwareInt8)
    
    sizes = {}
    for fname in os.listdir(output_dir):
        if fname.endswith('.onnx'):
            sizes[fname] = os.path.getsize(fname) / 1024 / 1024
    
    return sizes


def benchmark_onnx(model_path, model_type, audio_sample=None):
    """Benchmark ONNX vs PyTorch inference speed."""
    import time
    
    # Create dummy audio if not provided
    if audio_sample is None:
        sr = 16000
        audio_sample = np.random.randn(sr * 5).astype(np.float32)  # 5 seconds
    
    results = {}
    
    if model_type == 'whisper':
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        # Load ONNX
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        # Warmup
        inputs = processor(audio_sample, sampling_rate=16000, return_tensors="pt")
        _ = ort_model.generate(**inputs, max_new_tokens=128)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = ort_model.generate(**inputs, max_new_tokens=128)
            times.append(time.perf_counter() - start)
        
        results['onnx_latency'] = np.mean(times)
        results['onnx_latency_std'] = np.std(times)
    
    elif model_type == 'wav2vec2':
        from optimum.onnxruntime import ORTModelForCTC
        from transformers import Wav2Vec2Processor
        
        ort_model = ORTModelForCTC.from_pretrained(model_path)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        inputs = processor(audio_sample, sampling_rate=16000, return_tensors="pt")
        
        # Warmup
        _ = ort_model(**inputs)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = ort_model(**inputs)
            times.append(time.perf_counter() - start)
        
        results['onnx_latency'] = np.mean(times)
        results['onnx_latency_std'] = np.std(times)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['whisper', 'wav2vec2'], required=True)
    parser.add_argument('--quantize', action='store_true', default=True)
    parser.add_argument('--benchmark', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.model_path + '_onnx'
    
    print("=" * 50)
    print(f"ONNX Export: {args.model_type}")
    print("=" * 50)
    
    # Export
    if args.model_type == 'whisper':
        sizes = export_whisper_onnx(args.model_path, args.output_dir, args.quantize)
    else:
        sizes = export_wav2vec2_onnx(args.model_path, args.output_dir, args.quantize)
    
    print("\nModel sizes (MB):")
    for name, size in sizes.items():
        print(f"  {name}: {size:.2f} MB")
    
    # Benchmark
    if args.benchmark:
        print("\nBenchmarking...")
        bench = benchmark_onnx(args.output_dir, args.model_type)
        print(f"  Latency: {bench['onnx_latency']*1000:.1f} ± {bench['onnx_latency_std']*1000:.1f} ms")
    
    # Save metadata
    meta = {
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'model_type': args.model_type,
        'quantized': args.quantize,
        'sizes_mb': sizes,
        'benchmark': bench if args.benchmark else None
    }
    
    with open(os.path.join(args.output_dir, 'export_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

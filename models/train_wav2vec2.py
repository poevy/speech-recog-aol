"""
Wav2Vec2 Fine-Tuning Script for Code-Switching ASR
===================================================

This script fine-tunes indonesian-nlp/wav2vec2-indonesian-javanese-sundanese
on Indonesian-English code-switching data.

Usage:
    python train_wav2vec2.py --output_dir outputs/wav2vec2 --max_steps 2000

Key features:
- CTC-based training (different from Whisper's seq2seq)
- Character-level tokenization
- Wav2Vec2Processor for audio preprocessing
- Evaluation with WER/CER
"""

import os
import sys
import argparse
import json
import time
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    Wav2Vec2CTCTokenizer
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preparation import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Wav2Vec2 for code-switching ASR')
    
    parser.add_argument('--output_dir', type=str, default='outputs/wav2vec2',
                       help='Output directory')
    parser.add_argument('--model_name', type=str, 
                       default='indonesian-nlp/wav2vec2-indonesian-javanese-sundanese',
                       help='Pretrained model')
    parser.add_argument('--max_steps', type=int, default=2000,
                       help='Maximum training steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save frequency')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--freeze_feature_encoder', action='store_true', default=True,
                       help='Freeze feature encoder')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cs_train_size', type=int, default=1000)
    parser.add_argument('--cs_test_size', type=int, default=150)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    return parser.parse_args()


def prepare_dataset(batch, processor):
    """Prepare batch for Wav2Vec2 CTC training."""
    # Audio is already loaded at correct sample rate
    audio = batch["audio"]
    
    # Process audio (Wav2Vec2 uses raw waveform)
    batch["input_values"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # Tokenize text at character level
    with processor.as_target_processor():
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC training."""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        
        # Replace padding with -100 (ignore in loss)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        return batch


def compute_metrics(pred, processor, wer_metric, cer_metric):
    """Compute WER and CER."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Decode
    pred_str = processor.batch_decode(pred_ids)
    
    # Labels
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids)
    
    # Normalize
    pred_str = [s.lower().strip() for s in pred_str]
    label_str = [s.lower().strip() for s in label_str]
    
    # Compute metrics
    wer_value = wer_metric.compute(predictions=pred_str, references=label_str)
    cer_value = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer_value, "cer": cer_value}


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Wav2Vec2 Fine-Tuning for Code-Switching ASR")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)
    
    # Load processor and model
    print("\nLoading model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    
    # Freeze feature encoder if specified
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        print("Feature encoder frozen")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = get_datasets(
        cs_train_size=args.cs_train_size,
        cs_test_size=args.cs_test_size,
        seed=args.seed,
        verbose=True
    )
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = train_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=train_dataset.column_names,
        desc="Preparing train"
    )
    
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=eval_dataset.column_names,
        desc="Preparing eval"
    )
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Metrics
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=25,
        learning_rate=args.learning_rate,
        warmup_steps=min(100, args.max_steps // 10),
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        group_by_length=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, processor, wer_metric, cer_metric),
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print(f"\nTraining completed in {(time.time() - start_time) / 60:.1f} minutes")
    
    # Save
    print("\nSaving model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Final eval
    eval_results = trainer.evaluate()
    print("\nFinal Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")
    
    # Save results
    results = {
        "training_time_minutes": (time.time() - start_time) / 60,
        "eval_results": eval_results,
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()

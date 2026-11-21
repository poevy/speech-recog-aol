"""
Whisper-Tiny Fine-Tuning Script for Code-Switching ASR
=======================================================

This script fine-tunes openai/whisper-tiny on Indonesian-English code-switching data.

Usage:
    python train_whisper_tiny.py --output_dir outputs/whisper_tiny --max_steps 2000

Key features:
- Gradient checkpointing for memory efficiency
- Evaluation during training
- Early stopping based on validation WER
- Checkpoint saving
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)
from datasets import Audio
from jiwer import wer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preparation import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Whisper-tiny for code-switching ASR')
    
    parser.add_argument('--output_dir', type=str, default='outputs/whisper_tiny',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--model_name', type=str, default='openai/whisper-tiny',
                       help='Pretrained model to fine-tune')
    parser.add_argument('--max_steps', type=int, default=2000,
                       help='Maximum training steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Checkpoint save frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size per device')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cs_train_size', type=int, default=1000,
                       help='Number of synthetic code-switching training samples')
    parser.add_argument('--cs_test_size', type=int, default=150,
                       help='Number of synthetic code-switching test samples')
    parser.add_argument('--max_eval_samples', type=int, default=100,
                       help='Maximum samples for evaluation during training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume training from checkpoint path')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Weights & Biases project name')
    
    return parser.parse_args()


def prepare_dataset(batch, processor, feature_extractor):
    """
    Prepare batch for Whisper training.
    - Load and resample audio
    - Compute log-mel spectrogram (input_features)
    - Tokenize text (labels)
    """
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute log-mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize text
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


@torch.no_grad()
def compute_metrics(pred, processor, metric=wer):
    """
    Compute WER during evaluation.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize text
    pred_str = [s.lower().strip() for s in pred_str]
    label_str = [s.lower().strip() for s in label_str]
    
    # Compute WER
    wer_value = metric(label_str, pred_str)
    
    return {"wer": wer_value}


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("Whisper-Tiny Fine-Tuning for Code-Switching ASR")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Load processor and model
    print("\nLoading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="indonesian", task="transcribe")
    feature_extractor = processor.feature_extractor
    
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Override language and task tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")
    model.config.suppress_tokens = []
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = get_datasets(
        cs_train_size=args.cs_train_size,
        cs_test_size=args.cs_test_size,
        eval_size=args.max_eval_samples,
        seed=args.seed,
        verbose=True
    )
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = train_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor, "feature_extractor": feature_extractor},
        remove_columns=train_dataset.column_names,
        desc="Preparing training data"
    )
    
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor, "feature_extractor": feature_extractor},
        remove_columns=eval_dataset.column_names,
        desc="Preparing eval data"
    )
    
    # Data collator
    from transformers import WhisperDataCollatorWithPadding
    data_collator = WhisperDataCollatorWithPadding(processor=processor)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=25,
        learning_rate=args.learning_rate,
        warmup_steps=min(100, args.max_steps // 10),
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["wandb"] if args.wandb_project else ["none"],
        run_name=f"whisper-tiny-cs-{datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.wandb_project else None,
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\nFinal Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "training_time_minutes": training_time / 60,
        "eval_results": eval_results,
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {args.output_dir}/final")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

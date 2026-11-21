"""
Data Preparation Module for UKM Live Notulen ASR
Extracted from UKM_Live_Notulen_ASR_(2).ipynb

This module provides functions to:
- Load FLEURS Indonesian and English datasets
- Generate synthetic code-switching data
- Load custom UKM recordings (optional)
- Prepare train/eval/test splits for model training
"""

import os
import random
import numpy as np
from datasets import load_dataset, Audio, Dataset, concatenate_datasets, Features, Value
from typing import Optional, Tuple, Dict, Any

# Default configuration
DEFAULT_SR = 16000  # Sample rate
DEFAULT_SEED = 42


def load_fleurs(lang: str, split: str, sr: int = DEFAULT_SR, verbose: bool = True) -> Dataset:
    """
    Load FLEURS dataset for a specific language.
    
    Args:
        lang: Language code (e.g., 'id_id' for Indonesian, 'en_us' for English)
        split: Dataset split ('train', 'test', 'validation')
        sr: Sample rate for audio (default: 16000)
        verbose: Print progress messages
    
    Returns:
        Dataset with 'audio' and 'text' columns
    """
    # Try parquet branch first (faster, no loading script), fallback to repo default
    try:
        ds = load_dataset(
            "parquet", 
            split=split, 
            data_files={split: f"hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet/{lang}/{split}/*.parquet"}
        )
    except Exception as e:
        if verbose:
            print(f"Parquet loading failed, falling back: {repr(e)[:200]}")
        ds = load_dataset("google/fleurs", lang, split=split, trust_remote_code=True)
    
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    ds = ds.rename_column("raw_transcription", "text")
    return ds.select_columns(["audio", "text"])


def make_codeswitch(
    ds_id: Dataset, 
    ds_en: Dataset, 
    n: int, 
    seed: int = DEFAULT_SEED,
    max_sec: float = 28.0, 
    gap_sec: float = 0.15,
    sr: int = DEFAULT_SR,
    verbose: bool = True
) -> Dataset:
    """
    Generate synthetic code-switching dataset by combining Indonesian and English utterances.
    
    Strategy: [ID utterance] + pause + [EN utterance] (or vice versa)
    This helps the model learn language switching within audio without translation.
    
    Args:
        ds_id: Indonesian dataset
        ds_en: English dataset  
        n: Number of code-switched samples to generate
        seed: Random seed for reproducibility
        max_sec: Maximum total duration in seconds (Whisper limit is 30s)
        gap_sec: Silence gap between utterances in seconds
        sr: Sample rate
        verbose: Print progress messages
    
    Returns:
        Dataset with synthetic code-switched audio and combined transcripts
    """
    rng = random.Random(seed)
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    
    def gen():
        made, tries = 0, 0
        while made < n and tries < n * 5:
            tries += 1
            # Randomly select one Indonesian and one English utterance
            a = ds_id[rng.randrange(len(ds_id))]
            b = ds_en[rng.randrange(len(ds_en))]
            
            pa = (a["audio"]["array"].astype(np.float32), a["text"])
            pb = (b["audio"]["array"].astype(np.float32), b["text"])
            
            # 70% start with Indonesian, 30% start with English
            first, second = (pa, pb) if rng.random() < 0.7 else (pb, pa)
            
            # Concatenate audio with gap
            wav = np.concatenate([first[0], gap, second[0]])
            
            # Skip if too long (Whisper max is 30s)
            if len(wav) / sr > max_sec:
                continue
            
            made += 1
            yield {"audio": {"array": wav, "sampling_rate": sr}, "text": f"{first[1]} {second[1]}"}
    
    return Dataset.from_generator(
        gen, 
        features=Features({"audio": Audio(sampling_rate=sr), "text": Value("string")})
    )


def load_ukm_data(
    data_dir: str, 
    sr: int = DEFAULT_SEED,
    test_size: float = 0.3,
    seed: int = DEFAULT_SEED,
    verbose: bool = True
) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """
    Load custom UKM meeting recordings from local directory.
    
    Expected structure:
        data_dir/
        ├── *.wav (16 kHz mono, each clip <= 30 seconds)
        └── metadata.csv (columns: file_name, transcription)
    
    Args:
        data_dir: Path to UKM data directory
        sr: Sample rate
        test_size: Fraction for test split
        seed: Random seed
        verbose: Print progress messages
    
    Returns:
        Tuple of (train_dataset, test_dataset) or (None, None) if directory doesn't exist
    """
    import csv
    
    if not os.path.isdir(data_dir):
        if verbose:
            print(f"UKM data directory not found: {data_dir}")
        return None, None
    
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        if verbose:
            print(f"Metadata file not found: {metadata_path}")
        return None, None
    
    with open(metadata_path, encoding="utf-8") as f:
        meta = [r for r in csv.DictReader(f) if r["transcription"].strip()]
    
    if len(meta) == 0:
        if verbose:
            print("No valid entries in metadata.csv")
        return None, None
    
    own = Dataset.from_dict({
        "audio": [os.path.join(data_dir, r["file_name"]) for r in meta],
        "text": [r["transcription"].strip() for r in meta]
    }).cast_column("audio", Audio(sampling_rate=sr))
    
    # Split into train/test
    sp = own.train_test_split(test_size=test_size, seed=seed)
    
    if verbose:
        print(f"Loaded {len(sp['train'])} train and {len(sp['test'])} test UKM samples")
    
    return sp["train"], sp["test"]


def get_datasets(
    sr: int = DEFAULT_SR,
    seed: int = DEFAULT_SEED,
    cs_train_size: int = 1000,
    cs_test_size: int = 150,
    eval_size: int = 100,
    fleurs_id_test_size: int = 200,
    ukm_data_dir: Optional[str] = None,
    oversample_ukm: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Prepare all datasets for training and evaluation.
    
    Returns a dictionary containing:
        - train: Combined training dataset
        - eval: Evaluation dataset (subset of synthetic CS)
        - test_sets: Dict of test datasets by source
    
    Args:
        sr: Sample rate
        seed: Random seed
        cs_train_size: Number of synthetic code-switching samples for training
        cs_test_size: Number of synthetic code-switching samples for testing
        eval_size: Size of evaluation set during training
        fleurs_id_test_size: Number of FLEURS Indonesian test samples
        ukm_data_dir: Path to custom UKM recordings (optional)
        oversample_ukm: Factor to oversample UKM data (domain adaptation)
        verbose: Print progress messages
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if verbose:
        print("=" * 50)
        print("Loading datasets...")
        print("=" * 50)
    
    # 1. Load FLEURS Indonesian
    if verbose:
        print("\n[1/4] Loading FLEURS Indonesian...")
    id_train = load_fleurs("id_id", "train", sr=sr, verbose=verbose)
    id_test = load_fleurs("id_id", "test", sr=sr, verbose=verbose)
    
    # 2. Load FLEURS English (use validation for train to save download time)
    if verbose:
        print("\n[2/4] Loading FLEURS English...")
    en_train = load_fleurs("en_us", "validation", sr=sr, verbose=verbose)
    en_test = load_fleurs("en_us", "test", sr=sr, verbose=verbose)
    
    # 3. Generate synthetic code-switching
    if verbose:
        print("\n[3/4] Generating synthetic code-switching data...")
    cs_train = make_codeswitch(id_train, en_train, cs_train_size, seed=seed, sr=sr, verbose=verbose)
    cs_test = make_codeswitch(id_test, en_test, cs_test_size, seed=seed + 1, sr=sr, verbose=verbose)
    
    # 4. Load UKM data (optional)
    if verbose:
        print("\n[4/4] Loading UKM custom data...")
    own_train, own_test = None, None
    if ukm_data_dir:
        own_train, own_test = load_ukm_data(ukm_data_dir, sr=sr, seed=seed, verbose=verbose)
    
    # Combine training data
    if verbose:
        print("\n" + "=" * 50)
        print("Combining training datasets...")
    
    train_parts = [id_train, en_train, cs_train]
    
    # Oversample UKM data if available (important for domain adaptation)
    if own_train is not None:
        train_parts.extend([own_train] * oversample_ukm)
    
    train_ds = concatenate_datasets(train_parts).shuffle(seed=seed)
    
    # Prepare evaluation set (small subset for training-time evaluation)
    eval_ds = cs_test.select(range(min(eval_size, len(cs_test))))
    
    # Prepare test sets dictionary
    test_sets = {
        "fleurs_id": id_test.select(range(min(fleurs_id_test_size, len(id_test)))),
        "synthetic_cs": cs_test
    }
    
    if own_test is not None:
        test_sets["ukm_real"] = own_test
    
    if verbose:
        print(f"\nFinal dataset sizes:")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Eval:  {len(eval_ds)} samples")
        print(f"  Test sets: {', '.join(f'{k}({len(v)})' for k, v in test_sets.items())}")
        print("=" * 50)
    
    return {
        "train": train_ds,
        "eval": eval_ds,
        "test_sets": test_sets,
        "components": {
            "id_train": id_train,
            "id_test": id_test,
            "en_train": en_train,
            "en_test": en_test,
            "cs_train": cs_train,
            "cs_test": cs_test,
            "own_train": own_train,
            "own_test": own_test
        }
    }


def get_sample_batch(dataset: Dataset, n: int = 3) -> list:
    """
    Get a sample batch from dataset for inspection.
    
    Args:
        dataset: Dataset to sample from
        n: Number of samples
    
    Returns:
        List of samples with audio info and text
    """
    samples = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        samples.append({
            "index": i,
            "text": item["text"],
            "audio_length_sec": len(item["audio"]["array"]) / item["audio"]["sampling_rate"],
            "sample_rate": item["audio"]["sampling_rate"]
        })
    return samples


# Convenience function for quick testing
def quick_test():
    """Run a quick test of data loading with small subset."""
    print("Running quick test with small dataset sizes...")
    
    results = get_datasets(
        cs_train_size=10,  # Small for testing
        cs_test_size=5,
        eval_size=3,
        fleurs_id_test_size=5,
        verbose=True
    )
    
    print("\nSample from training set:")
    samples = get_sample_batch(results["train"], n=2)
    for s in samples:
        print(f"  [{s['index']}] ({s['audio_length_sec']:.2f}s): {s['text'][:100]}...")
    
    return results


if __name__ == "__main__":
    quick_test()

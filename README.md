# UKM Live Notulen - Code-Switching ASR

Indonesian-English Code-Switching Automatic Speech Recognition for Meeting Transcription.

## Project Structure

`
speech-recog-aol/
├── UKM_Live_Notulen_ASR_(2).ipynb  # Original notebook (Whisper-base)
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Dataset loading utilities
│   ├── metrics.py               # WER, CER, RTF, latency metrics
│   └── visualization.py         # Comparison charts/graphs
├── models/
│   ├── train_whisper_tiny.py    # Whisper-tiny training script
│   ├── train_wav2vec2.py        # Wav2Vec2 training script
│   └── export_onnx.py           # ONNX export utilities
├── notebooks/
│   └── model_comparison.ipynb   # Unified comparison notebook
└── outputs/
    ├── whisper_tiny/
    ├── wav2vec2/
    └── comparison_results/
`

## Models

| Model | Architecture | Size | Purpose |
|-------|-------------|------|---------|
| Whisper-base | Encoder-Decoder | 74M | Main model (browser deploy) |
| Whisper-tiny | Encoder-Decoder | 39M | Speed comparison |
| Wav2Vec2-Large-XLSR | CTC-based | 300M | Indonesian specialist |

## Setup

`ash
pip install transformers datasets optimum[onnxruntime] accelerate jiwer librosa soundfile scikit-learn sentencepiece
`

## Training

### Whisper-tiny
`ash
python models/train_whisper_tiny.py --output_dir outputs/whisper_tiny --max_steps 2000
`

### Wav2Vec2
`ash
python models/train_wav2vec2.py --output_dir outputs/wav2vec2 --max_steps 2000
`

## Evaluation

`ash
jupyter notebook notebooks/model_comparison.ipynb
`

## ONNX Export

`ash
# Export best model
python models/export_onnx.py --model_path outputs/whisper_tiny/final --model_type whisper
`

## Results

See outputs/comparison_results/ for evaluation metrics and visualizations.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.46+
- CUDA-capable GPU (T4 or better recommended)

## License

Educational project for COMP6822001 Speech Recognition course.

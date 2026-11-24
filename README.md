@
# UKM Live Notulen - Code-Switching ASR

Indonesian-English Code-Switching Automatic Speech Recognition for Meeting Transcription.

## Project Structure

```text
speech-recog-aol/
├── UKM_Live_Notulen_ASR_(2).ipynb
├── train_colab.ipynb
├── src/
│   ├── data_preparation.py
│   ├── metrics.py
│   └── visualization.py
├── models/
│   ├── train_whisper_tiny.py
│   ├── train_wav2vec2.py
│   └── export_onnx.py
├── notebooks/
│   └── model_comparison.ipynb
└── outputs/
    ├── whisper_tiny/
    ├── wav2vec2/
    └── comparison_results/
```

## Models

| Model | Architecture | Size | Purpose |
|-------|-------------|------|---------|
| Whisper-base | Encoder-Decoder | 74M | Main model |
| Whisper-tiny | Encoder-Decoder | 39M | Speed comparison |
| Wav2Vec2-Large-XLSR | CTC-based | 300M | Indonesian specialist |

## Setup

```bash
pip install transformers datasets optimum[onnxruntime] accelerate jiwer librosa soundfile scikit-learn sentencepiece evaluate
```

## Training

### Google Colab (Recommended)

1. Open train_colab.ipynb
2. Set GPU runtime
3. Run all cells

### Local Training

```bash
python models/train_whisper_tiny.py --output_dir outputs/whisper_tiny --max_steps 1500
python models/train_wav2vec2.py --output_dir outputs/wav2vec2 --max_steps 1500
```

## Evaluation

```bash
jupyter notebook notebooks/model_comparison.ipynb
```

## ONNX Export

```bash
python models/export_onnx.py --model_path outputs/whisper_tiny/final --model_type whisper
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.46+
- CUDA GPU recommended

## License

Educational project for COMP6822001 Speech Recognition course.
@

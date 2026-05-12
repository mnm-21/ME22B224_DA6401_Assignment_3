# DA6401 Assignment 3 — Transformer NMT

Implementation of the Transformer architecture ("Attention Is All You Need") for Neural Machine Translation from German to English using the Multi30k dataset.

## 🚀 Links
- **Weights & Biases Report:** [View Final Report](https://wandb.ai/mayank-chandak21-/da6401-a3/reports/DA6401_Assignment_3_ME22B224--VmlldzoxNjg1NTM1Ng?accessToken=c3d7bdg0om7od3u026fnh9vqcmhgbmjytnbfkxi8r05qeqvbbgizc10apkfg7lpo)
- **GitHub Repository:** [mnm-21/ME22B224_DA6401_Assignment_3](https://github.com/mnm-21/ME22B224_DA6401_Assignment_3.git)

## 📁 Project Structure
- **`model.py`**: Core Transformer architecture including Multi-Head Attention, Positional Encoding, and the full Encoder/Decoder stack.
- **`train.py`**: Main training pipeline, greedy decoding inference, and BLEU evaluation logic.
- **`dataset.py`**: Data loading utilities for the Multi30k dataset using SpaCy tokenization.
- **`lr_scheduler.py`**: Implementation of the Noam Scheduler (Linear Warmup + Inverse Square Root Decay).
- **`experiments.py`**: Unified script to run all 5 mandatory ablation experiments (LR scheduling, Scaling factors, Attention heatmaps, PE types, and Label Smoothing).
- **`translate_sample.py`**: A simple test script to load the best model and translate sample German sentences.
- **`src_vocab.json` / `tgt_vocab.json`**: Saved vocabulary mappings for reproducibility.
- **`best_checkpoint.pt`**: Best model weights used for final evaluation and report analysis.

## 🛠 Usage
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test translation**: `python translate_sample.py`

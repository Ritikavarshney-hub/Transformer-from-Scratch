# English → Sanskrit Neural Machine Translator

> Transformer-based Seq2Seq model trained on the Saamayik parallel corpus  
> PyTorch • BPE Tokenization • Beam Search Decoding

---

## 📖 Overview

This project implements a **Transformer architecture from scratch in PyTorch** to translate English sentences into Sanskrit. It uses the [Saamayik](https://github.com/sutej-pal/Saamayik) dataset — a curated parallel corpus of ~43,000 English–Sanskrit sentence pairs — and employs **Byte Pair Encoding (BPE)** tokenization to handle Sanskrit's complex, highly inflected morphology.

Everything — the model architecture, tokenizer, training loop, and inference engine — is implemented from first principles.No external ML Libraries are used.

---

## 📁 Project Structure

```
.
├── model.py          ← Transformer architecture (Encoder-Decoder)
├── bpetokenizer.py      ← BPE subword tokenizer 
├── dataset.py        ← PyTorch Dataset, DataLoader
├── train.py          ← Training loop
├── inference.py      ← Greedy & beam search decoding
├── checkpoints/      ← Saved weights & vocabularies ( not in repo)
│   ├── best_model.pt
│   ├── src_vocab.json
│   └── tgt_vocab.json
└── data/
    └── final_data/
        ├── train.en / train.sa   (43,493 pairs)
        ├── dev.en   / dev.sa     (2,416 pairs)
        └── test.en  / test.sa    (2,417 pairs)
```

---

## 🏗️ Model Architecture

The model follows the original Transformer from *"Attention Is All You Need"* (Vaswani et al., 2017).

| Component | Description |
|-----------|-------------|
| `InputEmbeddings` | Token embedding scaled by √d_model |
| `PositionalEncoding` | Sinusoidal position encodings added to embeddings |
| `MultiHeadAttention` | Scaled dot-product attention split across h heads |
| `FeedForwardBlock` | Two-layer MLP with ReLU: d_model → d_ff → d_model |
| `LayerNormalisation` | Pre-norm applied before each sub-layer |
| `ResidualConnection` | Add & Norm: x + dropout(sublayer(norm(x))) |
| `EncoderLayer` | Self-attention + FFN with residual connections |
| `DecoderLayer` | Masked self-attention + cross-attention + FFN |
| `ProjectionLayer` | Linear projection to target vocabulary logits |

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Model / embedding dimension |
| `d_ff` | 1024 | Feed-forward inner dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 4 | Encoder and decoder layers each |
| `dropout` | 0.1 | Dropout rate |
| `max_len` | 150 | Maximum sequence length |
| `batch_size` | 32 | Training batch size |
| `warmup_steps` | 4000 | LR warmup steps |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |

---

## 🔤 Tokenizer — BPE Subword

The tokenizer uses **Byte Pair Encoding (BPE)** implemented from scratch with no external libraries.

BPE handles Sanskrit's complex inflectional morphology by breaking rare words into meaningful subword units — ensuring **zero unknown tokens** at inference time.

**How it works:**
1. Every word is split into individual characters, with a `▁` prefix to mark word boundaries
2. The most frequent adjacent character pair is merged into a single token
3. This merge step is repeated N times (4000 for English, 6000 for Sanskrit)
4. At inference, any unseen word is broken into known subwords, falling back to characters

**Example:**
```
Word-level:  "पाठयति"  →  <unk>              (unseen word = unknown)
BPE:         "पाठयति"  →  "▁पाठ" + "यति"    (known subwords)
```

---

## ⚙️ Setup & Installation

**Requirements:** Python 3.8+, PyTorch (CUDA recommended)

### Install PyTorch with CUDA 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU
```python
import torch
print(torch.cuda.is_available())    
print(torch.cuda.get_device_name(0))  
```

---

## 🚀 Training

```bash
python train.py
```

Vocabularies are built automatically on the first run and cached to `checkpoints/`. The best model by validation loss is saved to `checkpoints/best_model.pt`.

### Expected Training Progress

| Val Loss | Translation Quality |
|----------|---------------------|
| Above 4.0 | Mostly gibberish |
| 3.0 – 4.0 | Wrong but recognizable words |
| 2.5 – 3.0 | Partially correct sentences |
| 2.0 – 2.5 | Mostly correct grammar |
| Below 2.0 | Good translations |

---

## 🔍 Inference

**Interactive mode:**
```bash
python inference.py
```

**Translate a file:**
```bash
python inference.py --input sentences.txt --output translations.txt
```

## 📊 Dataset — Saamayik

| Split | Sentence Pairs | Usage |
|-------|---------------|-------|
| Train | 43,493 | Model training |
| Dev | 2,416 | Validation during training |
| Test | 2,417 | Final evaluation |

---

*Built with PyTorch • Saamayik Dataset • Transformer (Vaswani et al., 2017)*

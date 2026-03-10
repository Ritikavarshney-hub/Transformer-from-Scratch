# English → Sanskrit Machine Translator

A Transformer-based sequence-to-sequence model trained on the **Saamayik** parallel corpus.

---

## Project Structure

```
.
├── model.py          ← Transformer architecture (fixed)
├── tokenizer.py      ← Word-level tokenizer + Vocabulary
├── dataset.py        ← PyTorch Dataset & DataLoader
├── train.py          ← Training loop
├── inference.py      ← Greedy & beam-search decoding
├── checkpoints/      ← Saved model weights & vocabularies (auto-created)
└── data/
    └── final_data/
        ├── train.en / train.sa   (43 k pairs)
        ├── dev.en   / dev.sa     (2.4 k pairs)
        └── test.en  / test.sa    (2.4 k pairs)
```

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (use the CUDA wheel if you have a GPU)
```

No other external libraries are required.

---

## Training

```bash
python train.py
```

Edit the `CONFIG` dict at the top of `train.py` to adjust hyperparameters:

| Key | Default | Description |
|-----|---------|-------------|
| `d_model` | 256 | Model dimension |
| `d_ff` | 1024 | Feed-forward inner dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 4 | Encoder/decoder layers |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 30 | Number of epochs |
| `max_len` | 150 | Max token sequence length |
| `label_smoothing` | 0.1 | Label smoothing for cross-entropy |

The best model (lowest validation loss) is saved to `checkpoints/best_model.pt`.

---

## Inference

**Interactive mode:**
```bash
python inference.py
```

**Translate a file:**
```bash
python inference.py --input my_sentences.txt --output translations.txt
```

**Options:**
```
--checkpoint   Path to model checkpoint  (default: checkpoints/best_model.pt)
--method       greedy | beam             (default: beam)
--beam_size    Beam width                (default: 5)
```

---

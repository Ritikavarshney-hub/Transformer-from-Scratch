# English-to-Sanskrit Neural Machine Translation using Transformer (From Scratch)

## Overview

This project implements an **English-to-Sanskrit Neural Machine Translation (NMT)** system by building the **Transformer architecture from scratch in PyTorch**, without using `torch.nn.Transformer`. The project includes the complete machine translation pipeline, from dataset preparation and custom tokenization to model training, inference, and evaluation.

The model is trained on a parallel English–Sanskrit corpus collected from multiple publicly available datasets and evaluated using standard machine translation metrics.

---

## Features

* Transformer architecture implemented completely from scratch
* Custom Byte Pair Encoding (BPE) tokenizer
* Multi-Head Self Attention
* Positional Encoding
* Encoder–Decoder architecture
* Label Smoothing
* Learning Rate Warmup Scheduler
* Mixed Precision (FP16) Training
* Gradient Clipping
* Early Stopping
* Beam Search Decoding
* BLEU and BERTScore evaluation

---

## Model Architecture

| Component               | Configuration |
| ----------------------- | ------------- |
| Encoder Layers          | 4             |
| Decoder Layers          | 4             |
| Attention Heads         | 8             |
| Embedding Dimension     | 256           |
| Feed Forward Dimension  | 1024          |
| Maximum Sequence Length | 150           |
| Dropout                 | 0.1           |
| Training Epochs         | 50            |
| Effective Batch Size    | 128           |

---

## Dataset

The model is trained on a merged English–Sanskrit parallel corpus created from multiple datasets including:

* Bible
* Gita Sopanam
* MKB
* NIOS
* Spoken Tutorials
* Itihasa

### Dataset Statistics

| Split      | Sentence Pairs |
| ---------- | -------------: |
| Training   |         43,493 |
| Validation |          2,416 |
| Test       |          2,417 |
| **Total**  |     **48,326** |

---

## Project Structure

```
.
├── data/
│   ├── final_data/
│   ├── bible/
│   ├── gitasopanam/
│   ├── mkb/
│   ├── nios/
│   ├── spoken-tutorials/
│   └── itihasa/
│
├── checkpoints/
│   ├── best_model.pt
│   ├── src_vocab.json
│   └── tgt_vocab.json
│
├── bpetokenizer.py
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── evaluate.py
└── run_test.py
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd English-to-sanskrit-transformer
```

Install dependencies

```bash
pip install torch numpy pandas sacrebleu bert-score tqdm sentencepiece
```

---

## Training

Train the model

```bash
python train.py
```

The training pipeline includes:

* Dataset preprocessing
* Vocabulary generation
* BPE tokenization
* Mixed precision training
* Checkpoint saving
* Validation after every epoch
* Early stopping

The best model is automatically stored inside the `checkpoints/` directory.

---

## Inference

Translate an English sentence

```bash
python inference.py
```

The inference module supports:

* Greedy Decoding
* Beam Search Decoding

---

## Evaluation

Run evaluation on the test set

```bash
python evaluate.py
```

The evaluation script computes:

* Corpus BLEU Score (SacreBLEU)
* BERTScore Precision
* BERTScore Recall
* BERTScore F1

---

## Results

Evaluation was performed on **2,417 unseen English–Sanskrit sentence pairs** using **Beam Search (Beam Size = 5)**.

| Metric              |      Score |
| ------------------- | ---------: |
| BLEU                |  **12.36** |
| BERTScore Precision | **0.7856** |
| BERTScore Recall    | **0.7727** |
| BERTScore F1        | **0.7787** |

---

## Technologies Used

* Python
* PyTorch
* Transformer Architecture
* Byte Pair Encoding (BPE)
* Multi-Head Attention
* Beam Search
* Mixed Precision (FP16)
* SacreBLEU
* BERTScore
* NumPy
* Pandas

---

## Key Learning Outcomes

* Implemented the complete Transformer architecture without relying on high-level Transformer APIs.
* Built a custom tokenization and preprocessing pipeline for bilingual machine translation.
* Learned sequence-to-sequence learning, attention mechanisms, positional encoding, and beam search decoding.
* Evaluated translation quality using standard machine translation metrics such as BLEU and BERTScore.
* Gained practical experience in training large neural networks using mixed precision, checkpointing, and learning-rate scheduling.

---

## Future Improvements

* Train on larger English–Sanskrit corpora
* Increase model depth and embedding dimensions
* Use pretrained multilingual embeddings
* Add Transformer variants such as T5 or mBART for comparison
* Deploy the model as a web application using FastAPI or Streamlit
* Perform hyperparameter optimization for improved translation quality

---

## Author

Developed as a deep learning project to explore Transformer-based Neural Machine Translation for English-to-Sanskrit translation using PyTorch.

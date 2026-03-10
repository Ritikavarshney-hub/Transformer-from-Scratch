#English → Sanskrit Neural Machine Translator
Transformer-based Seq2Seq model trained on the Saamayik parallel corpus
PyTorch  •  BPE Tokenization  •  Beam Search Decoding  •  RTX 3050 Trained

📖 Project Overview
This project implements a Transformer architecture from scratch in PyTorch to translate English sentences into Sanskrit. It uses the Saamayik dataset — a curated parallel corpus of ~43,000 English-Sanskrit sentence pairs — and employs Byte Pair Encoding (BPE) tokenization to handle Sanskrit's complex, highly inflected morphology.
The project contains no external ML dependencies beyond PyTorch. Everything — the model architecture, tokenizer, training loop, and inference engine — is implemented from first principles.

📁 Project Structure
.
├── model.py          ← Transformer architecture (Encoder-Decoder)
├── tokenizer.py      ← BPE subword tokenizer (no external libs)
├── dataset.py        ← PyTorch Dataset, DataLoader, masking
├── train.py          ← Training loop with warmup LR scheduler
├── inference.py      ← Greedy & beam search decoding
├── checkpoints/      ← Saved weights & vocabularies (auto-created)
│   ├── best_model.pt
│   ├── src_vocab.json
│   └── tgt_vocab.json
└── data/
    └── final_data/
        ├── train.en / train.sa   (43,493 pairs)
        ├── dev.en   / dev.sa     (2,416 pairs)
        └── test.en  / test.sa    (2,417 pairs)

🏗️ Model Architecture
The model follows the original Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) with an Encoder-Decoder structure.
Components
Component	Description
InputEmbeddings	Token embedding scaled by √d_model
PositionalEncoding	Sinusoidal position encodings added to embeddings
MultiHeadAttention	Scaled dot-product attention split across h heads
FeedForwardBlock	Two-layer MLP with ReLU: d_model → d_ff → d_model
LayerNormalisation	Pre-norm applied before each sub-layer
ResidualConnection	Add & Norm: x + dropout(sublayer(norm(x)))
EncoderLayer	Self-attention + FFN with residual connections
DecoderLayer	Masked self-attention + cross-attention + FFN
ProjectionLayer	Linear projection to target vocabulary logits

Default Hyperparameters
Parameter	Value	Description
d_model	256	Model / embedding dimension
d_ff	1024	Feed-forward inner dimension
num_heads	8	Attention heads
num_layers	4	Encoder and decoder layers each
dropout	0.1	Dropout rate
max_len	150	Maximum sequence length
batch_size	32	Training batch size
warmup_steps	4000	LR warmup steps
label_smoothing	0.1	Cross-entropy label smoothing


🔤 Tokenizer — BPE Subword
The tokenizer uses Byte Pair Encoding (BPE) — the same technique used by GPT models — implemented from scratch with no external libraries.
Unlike word-level tokenization, BPE handles Sanskrit's complex inflectional morphology by breaking rare words into meaningful subword units, ensuring zero unknown tokens at inference time.
How It Works
•Every word is split into individual characters, with a ▁ prefix to mark word boundaries
•The most frequent adjacent character pair is merged into a single token
•This merge step is repeated N times (4000 for English, 6000 for Sanskrit)
•At inference, any unseen word is broken into known subwords, falling back to characters

Example
Word-level:  "पाठयति"  →  <unk>   (unseen word = unknown)
BPE:         "पाठयति"  →  "▁पाठ" + "यति"   (known subwords)

⚙️ Setup & Installation
Requirements
•Python 3.8+
•PyTorch (CUDA recommended — see below)
•No other ML libraries required

Install PyTorch with CUDA (RTX 3050 / CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Verify GPU
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 3050

🚀 Training
Start Training
python train.py
Vocabularies are built automatically on the first run and cached to checkpoints/. The best model by validation loss is saved to checkpoints/best_model.pt.
Resume from Checkpoint
Add this block to train.py just before the training loop to resume instead of starting over:
if os.path.exists(CONFIG["best_model"]):
    ckpt = torch.load(CONFIG["best_model"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"Resumed from epoch {ckpt['epoch']}")
Monitor GPU Usage
nvidia-smi -l 2
GPU-Util should read 50–99% during training. Memory usage will be around 1–3 GB.
Expected Training Progress
Val Loss	Perplexity	Translation Quality
Above 4.0	55+	Mostly gibberish
3.0 – 4.0	20 – 55	Wrong but recognizable words
2.5 – 3.0	12 – 20	Partially correct sentences
2.0 – 2.5	7 – 12	Mostly correct grammar
Below 2.0	< 7	Good translations


🔍 Inference
Interactive Mode
python inference.py
Translate a File
python inference.py --input sentences.txt --output translations.txt
Options
Flag	Default	Description
--checkpoint	checkpoints/best_model.pt	Path to trained model weights
--method	beam	Decoding method: greedy or beam
--beam_size	5	Beam width for beam search
--input	None	Input text file (one sentence per line)
--output	None	Output file for translations


📊 Dataset — Saamayik
The model is trained on the Saamayik parallel corpus, a curated English-Sanskrit dataset compiled from multiple sources.
Split	Sentence Pairs	Usage
Train	43,493	Model training
Dev	2,416	Validation during training
Test	2,417	Final evaluation


🐛 Bugs Fixed in Original model.py
Location	Bug	Fix
PositionalEncoding.forward	.requires_grad(False) is not a valid tensor method	Changed to .detach()
MultiHeadAttention.forward	Batch dim used query.shape[1] instead of shape[0]	Fixed to query.shape[0] and -1 for seq dim
MultiHeadAttention.attention	Called as instance method; self passed as query	Added @staticmethod and fixed call site
ProjectionLayer.forward	log_softmax applied before CrossEntropyLoss (double softmax)	Returns raw logits; loss handles softmax
build_transformer defaults	d_model=512, layers=6 causes overfitting on 43k pairs	Reduced to d_model=256, layers=4


💡 Tips for Better Results
•Train for 60–100 epochs — 30 epochs is not enough for good translations
•Use a larger model on GPU: d_model=512, d_ff=2048, num_layers=6
•Increase batch_size to 64 on GPU to speed up training
•Add the Bible and Gitasopanam sub-corpora from the Saamayik zip for more data
•If loss plateaus, try reducing dropout to 0.05
•Beam search (beam_size=5) gives noticeably better results than greedy decoding

💾 Checkpoints & GitHub
Model weights (.pt files) are excluded from the repository via .gitignore because GitHub has a 100MB file size limit. The vocabulary JSON files (src_vocab.json, tgt_vocab.json) are small and safe to commit.
To share trained weights, use Git LFS, Hugging Face Hub, or Google Drive.
# .gitignore
checkpoints/*.pt
__pycache__/
*.pyc
data/
.env

Built with PyTorch  •  Saamayik Dataset  •  Transformer (Vaswani et al., 2017)

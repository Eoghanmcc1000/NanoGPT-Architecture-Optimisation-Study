# NanoGPT Architecture Optimisation Study

A comprehensive implementation and analysis of transformer language models, exploring the effects of model downsising, bias configurations, and skip connections on performance across different text domains.

## Project Overview

This project implements a character-level GPT-style transformer from scratch in PyTorch, conducting systematic experiments on model architecture components. The study evaluates how different hyperparameter configurations affect model performance when trained on child speech data and tested across diverse linguistic domains.

## Key Findings

### Model Performance Comparison

| Configuration | Parameters | Train Loss | Val Loss | Specialisation |
|---------------|------------|------------|----------|----------------|
| **Original** | 10.77M | 0.3285 | 0.3363 | Highest quality text generation |
| **Config 1** | 0.99M | 0.3374 | 0.3399 | **Optimal efficiency/performance balance** |
| **Config 2** | 0.93M | 0.3389 | 0.3433 | Deeper representations per layer |
| **Config 3** | 0.75M | 0.3451 | 0.3478 | Minimal computational load |

### Cross-Domain Evaluation

| Test Dataset | Best Model Loss | Dummy Model Loss | Performance |
|--------------|----------------|------------------|-------------|
| **Child Speech** | 0.3466 | 4.1940 | Excellent (12x better) |
| **Shakespeare** | 8.12 | 4.1944 | Poor (2x worse) |

## Architecture Components Studied

### Model Downsising Analysis
- **Original**: 384 embedding dim, 6 heads, 6 layers → 10.77M parameters
- **Optimised**: 140 embedding dim, 4 heads, 4 layers → 0.99M parameters
- **Result**: 91% parameter reduction with minimal performance loss

### Bias Configuration Experiments
Tested all combinations of bias in key/query/value projections:
- **Best Configuration**: Query + Value bias (Config 5)
- **Insight**: Full bias inclusion shows marginal improvement over no bias
- **Trade-off**: 16,800 additional parameters for minimal gain

### Skip Connection Impact
| Component | With Skip Connections | Without Skip Connections |
|-----------|----------------------|--------------------------|
| **Final Loss** | 0.3383 | 3.1268 |
| **Text Quality** | Coherent sentences | Completely jumbled |
| **Learning** | Rapid convergence | Severe degradation |

## Quick Start

### Requirements
```bash
# Core dependencies
torch
torch.nn
```

### Basic Usage
```bash
# Train the model
python transformer_model.py

# Generate text samples
python transformer_model.py --generate

# Evaluate on test sets
python transformer_model.py --evaluate
```

### Model Configuration
```python
# Optimal configuration found in study
n_embd = 140      # Embedding dimension
n_head = 4        # Number of attention heads  
n_layer = 4       # Number of transformer blocks
dropout = 0.2     # Dropout rate
```

## Project Structure

### Core Implementation
- `transformer_model.py` - Complete transformer implementation with all experimental configurations
- `datasets/` - Child speech and Shakespeare text data

### Model Components
- `Head` - Single self-attention head implementation
- `MultiHeadAttention` - Parallel attention mechanism
- `FeedForward` - Position-wise feed-forward networks
- `Block` - Complete transformer block with skip connections
- `GPTLanguageModel` - Full model architecture

### Experimental Configurations
- **Part A-C**: Model downsising experiments (1000 iterations)
- **Part D**: Bias configuration analysis across all projection matrices
- **Part E**: Skip connection ablation study
- **Part 2**: Extended training (5000 iterations) and cross-domain evaluation

## Experimental Methodology

### Training Protocol
- **Batch Size**: 64 sequences
- **Context Length**: 256 tokens
- **Optimiser**: AdamW with 3e-4 learning rate
- **Training Duration**: 1000-5000 iterations
- **Evaluation**: Train/validation split with periodic loss estimation

### Novel Features
- **Unknown Token Handling**: Robust character encoding for cross-domain evaluation
- **Systematic Bias Study**: Complete ablation across all attention projections
- **Architecture Comparison**: Skip connection impact analysis
- **Cross-Domain Testing**: Evaluation on linguistically diverse datasets

## Results Analysis

### Domain Specialisation
The model demonstrates strong **domain-specific learning**, achieving outstanding performance on child speech (similar to training data) whilst struggling with Shakespearean English. This highlights the importance of training data diversity for general-purpose language models.

### Architecture Insights
1. **Skip Connections**: Absolutely critical for training stability and text quality
2. **Model Downsising**: 91% parameter reduction possible with careful hyperparameter selection
3. **Bias Terms**: Minimal impact on performance; Query+Value bias slightly optimal
4. **Computational Efficiency**: Dramatic improvements possible through thoughtful architecture choices

### Practical Applications
- **Targeted Deployment**: Educational tools for children's language processing
- **Model Efficiency**: Demonstrates viable approaches for resource-constrained environments
- **Architecture Guidelines**: Provides empirical evidence for transformer design decisions

## Implementation Details

### Character-Level Tokenisation
```python
# Vocabulary handling with unknown token support
chars = sorted(list(set(text)))
chars.append('<unk>')
encode = lambda s: [stoi.get(c, unk_idx) for c in s]
```

### Attention Mechanism
- Scaled dot-product attention with causal masking
- Configurable bias terms in key/query/value projections
- Dropout regularisation for training stability

### Training Loop
- Periodic loss evaluation on train/validation splits
- AdamW optimisation with gradient clipping
- Cross-entropy loss for next-token prediction

## References

- **Karpathy, A.** (2023) *Let's build GPT: from scratch, in code, spelled out*
- **Attention Mechanism**: Vaswani et al. "Attention Is All You Need"
- **Architecture Inspiration**: GPT-style decoder-only transformer

---

*This implementation demonstrates the fundamental principles of transformer architectures through systematic experimentation and empirical analysis.*

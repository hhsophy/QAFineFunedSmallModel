# Learning Deep Learning

An educational project focused on learning how to train and fine-tune open-source models using PyTorch. The goal is to understand the concepts and implementation details behind model architectures, training pipelines, optimization, checkpointing, and efficient model development through hands-on experimentation.

## Goals

- Learn how neural network architectures work and how to implement them.
- Practice training and fine-tuning open-source models.
- Understand training loops, loss functions, optimizers, and learning-rate schedulers.
- Explore checkpointing and resuming training.
- Investigate GPU memory usage and training efficiency.
- Document experiments, observations, and lessons learned.

## Technologies

- Python
- PyTorch
- Open-source models and datasets

This project does not require Hugging Face Transformers. Implementations and tools may vary between experiments.

## Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd learning-deep-learning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies required by each experiment. For PyTorch installation instructions, visit [pytorch.org](https://pytorch.org/).

Follow the instructions in the relevant project folder or notebook, as requirements may differ between experiments.

## Project Structure

The repository structure will evolve as the project grows. It may include:

- Model implementations and experiments
- Training and fine-tuning scripts
- Dataset preparation utilities
- Checkpoint saving and resuming utilities
- Notes and examples about model training

## Educational Purpose

This repository is intended for learning, experimentation, and understanding the fundamentals of deep learning. It is a work in progress and is not intended to be a production-ready training framework.

Before reusing any third-party model, dataset, or asset, check its original license and usage terms.

## License

Original code in this repository is licensed under the MIT License unless otherwise specified. See [LICENSE](LICENSE) for details. Third-party models, datasets, and other assets may have separate license terms.
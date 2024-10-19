# SmolLM: Implementation, Fine-Tuning, and Alignment for Grammatical Error Correction

## Table of Contents

- [Introduction](#introduction)
- [Model Overview](#model-overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
  - [Custom SmolLM Implementation](#custom-smollm-implementation)
  - [Testing the Model](#testing-the-model)
  - [Fine-Tuning for Grammatical Error Correction](#fine-tuning-for-grammatical-error-correction)
  - [Creating a Preference Optimization Dataset](#creating-a-preference-optimization-dataset)
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Welcome to the **SmolLM** repository! This project demonstrates the implementation, fine-tuning, and alignment of the **SmolLM-135M** language model for **Grammatical Error Correction (GEC)** using the **Grammarly CoEdIT** dataset. The workflow encompasses:

1. **Custom Implementation**: Building the SmolLM-135M model architecture.
2. **Fine-Tuning**: Training the model on GEC tasks.
3. **Alignment**: Enhancing the model's performance using Direct Preference Optimization (DPO).

This repository includes a Colab notebook that guides you through each of these steps.

---

## Model Overview

- **SmolLM-135M**: A compact language model designed for efficient text generation. Available on [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM-135M).

---

## Prerequisites

### Hardware

- **GPU**: A single A100 NVIDIA GPU with 40GB memory is recommended for accelerated fine-tuning and alignment.

### Software

- **Python**: Version 3.12
- **Libraries**:
  - `torch`
  - `transformers`
  - `datasets`
  - `evaluate`
  - `trl`
  - `tqdm`
  - `fast_edit_distance`
  - `git` with `git-lfs`

### Installation

You can install the required Python packages using:

```bash
pip install torch transformers datasets evaluate trl tqdm fast_edit_distance git-lfs
```

---

## Setup

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/HassanMahmoudd/SmolLM_RL.git
   cd SmolLM_RL
   ```

2. **Install Git Large File Storage (LFS)**:
   
   ```bash
   git lfs install
   ```

3. **Clone Pre-trained Model Weights**:
   
   ```bash
   git clone https://huggingface.co/dsouzadaniel/C4AI_SMOLLM135
   mv C4AI_SMOLLM135/BareBones_SmolLM-135M.pt ./
   ```

4. **Verify Files**:
   
   ```bash
   ls
   ```

5. **Open the Colab Notebook**:
   
   Navigate to the notebook available on the repository or on Colab at [SmolLM_RL.ipynb](https://colab.research.google.com/drive/1_fUoVdpOiojMTGrRHznIRajH1R9rVEwx), make a copy and follow the step-by-step instructions.

---

## Usage

### Custom SmolLM Implementation

The notebook begins by defining the architecture of the SmolLM-135M model, including components like rotary embeddings, KV cache, multi-layer perceptrons (MLP), normalization layers, and attention mechanisms. It then loads the pre-trained weights to initialize the model.

### Testing the Model

Helper functions are provided to generate text and compare the outputs of the custom implementation with a reference model to ensure correctness.

### Fine-Tuning for Grammatical Error Correction

1. **Dataset Preparation**:
   - Loads the **Grammarly CoEdIT** dataset. Available on [HuggingFace](https://huggingface.co/datasets/grammarly/coedit).
   - Filters the dataset for GEC tasks.

2. **Model and Tokenizer Setup**:
   - Loads the pre-trained SmolLM-135M model and tokenizer.
   - Configures padding and truncation settings.

3. **Training**:
   - Formats and tokenizes the data.
   - Configures training parameters using `SFTConfig`.
   - Fine-tunes the model using `SFTTrainer`.

4. **Inference and Evaluation**:
   - Defines functions for performing inference on new sentences.
   - Evaluates the model's performance using BLEU scores.

### Creating a Preference Optimization Dataset

To enhance model performance, the notebook demonstrates how to create a preference optimization dataset by generating multiple output variants and annotating them based on their similarity to ground truth corrections.

### Direct Preference Optimization (DPO)

The final step involves applying Direct Preference Optimization to further train the model using the preference optimization dataset, aiming to align the model's outputs more closely with desired corrections.

---

## Conclusion

This project provides a comprehensive workflow for implementing, fine-tuning, and aligning the SmolLM-135M model for grammatical error correction tasks. By following the Colab notebook, you can replicate the process and adapt it to your specific needs, leveraging both supervised fine-tuning and preference optimization techniques to enhance model performance.

---

## References

- [SmolLM-135M on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [Grammarly CoEdIT Dataset](https://huggingface.co/datasets/grammarly/coedit)
- [TRL (Transformers Reinforcement Learning)](https://huggingface.co/docs/trl/en/index)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or suggestions, please open an issue or contact [hassanmahmoudsd@example.com](mailto:hassanmahmoudsd@example.com).
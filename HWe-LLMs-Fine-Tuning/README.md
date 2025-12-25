# LLM Fine-Tuning for Persian Dialogue

This repository contains the implementation and analysis of fine-tuning a Large Language Model (LLM) for **Persian conversational tasks**, conducted as part of the *Neural Networks and Deep Learning* course at the University of Tehran.

The project focuses on **instruction fine-tuning under limited computational resources**, using **parameter-efficient fine-tuning (PEFT)** methods and qualitative evaluation of model behavior.

---

## Project Overview

The main goal of this project is to study and implement different approaches for adapting a pre-trained LLM to Persian dialogue data, while minimizing computational and memory costs.

The following fine-tuning strategies are explored and compared:

- **Soft Prompt Tuning**
- **LoRA (Low-Rank Adaptation)**
- **Partial Full Fine-Tuning** (unfreezing first and last layers)

All experiments are performed without using end-to-end automated fine-tuning pipelines, in accordance with the course constraints.

---

## Dataset

- **SlimOrca (Persian-translated subset)**  
- A conversational instruction-style dataset derived from OpenOrca, translated to Persian.
- Each sample consists of structured messages with explicit `user` and `assistant` roles.

### Preprocessing Steps
- Inspection and filtering of conversation roles
- Conversion to chat-style training format
- Tokenization using the selected model tokenizer
- Alignment with model-specific chat templates

---

## Model

- **Base Model:** LLaMA-3.2-3B-Instruct  
- Chosen due to its instruction-following capabilities and compatibility with conversational fine-tuning.

The model is loaded using **4-bit quantization** to reduce memory usage and enable training on limited hardware.

---

## Fine-Tuning Methods

### 1. Soft Prompt Tuning
- Trainable virtual tokens are prepended to model inputs.
- The base model weights remain frozen.
- Implemented using HuggingFace `transformers` and `peft`.

### 2. LoRA (Low-Rank Adaptation)
- Low-rank adapters applied mainly to attention projection layers.
- Implemented using the `unsloth` library for improved speed and memory efficiency.
- Only LoRA parameters are updated during training.

### 3. Partial Fine-Tuning
- Only the first and last layers of the model are unfrozen.
- All other layers remain frozen.
- Used to compare traditional fine-tuning against PEFT approaches.

---

## Training Setup

- Frameworks: `HuggingFace Transformers`, `PEFT`, `TRL`
- Mixed precision training (FP16 / BF16 where supported)
- Gradient accumulation to handle small batch sizes
- Training progress monitored via loss curves

---

## Evaluation

Evaluation is primarily **qualitative**, focusing on:

- Relevance of responses
- Instruction-following behavior
- Coherence and completeness of generated Persian text

Model outputs before and after fine-tuning are compared on the same prompts to assess improvements.

---

## Key Observations

- PEFT methods significantly reduce memory and computation requirements.
- LoRA achieves better adaptation quality compared to soft prompts under similar constraints.
- Fine-tuning improves response relevance, though issues such as mixed-language tokens and occasional incoherence remain.
- Output quality is sensitive to maximum sequence length and dataset formatting.

# README: SAGE14-FX - Philosophical Meta-AGI

## Overview:
SAGE14-FX is an experimental model architecture inspired by the triangle of AGI: Memory, Pain, and Choice. It builds upon symbolic-situational reasoning concepts introduced in Sage14, but extends them with episodic memory, adaptive error sensitivity (pain), and learned hypothesis selection (choice).

## Core Concepts:
- EpisodicMemory: Stores task-specific embeddings across a few examples to simulate memory and context retention.
- TaskPainSystem: Tracks divergence between predictions and true output, adjusting sensitivity dynamically. This encodes the capacity to suffer.
- ChoiceHypothesisModule: Generates and selects among transformation hypotheses. This encodes deliberation.

## Architecture:
- Encoder: CNN-based perceptual feature extractor.
- Agent: GRU-based temporal integrator over training examples.
- Memory: Buffers hidden states during training phase.
- Chooser: Generates possible output embeddings.
- Decoder: Maps output embeddings to logits over grid colors.

## Training Setup:
- Input: Sequence of input-output example pairs from a single ARC task (few-shot setup)
- Output: Prediction for a held-out input sample
- Loss: Standard categorical crossentropy, plus optional auxiliary signals (task pain)

## Usage:
- Instantiate Sage14FX(hidden_dim=64)
- Input shape: (batch_size, sequence_length, 20, 20, 10)
- Target shape: (batch_size, sequence_length, 20, 20) [optional]
- Call: output_logits = model(x_seq, y_seq)

## Limitations:
- Only models single-task adaptation; does not yet support multi-task inference.
- Requires aligned input/output formatting across few-shot samples.

## Future Directions:
- Introduce symbolic sketching modules
- Add meta-loss to optimize over task success
- Implement pain-based gradient gating

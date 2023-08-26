# Brain-Computer-Development
# Eye-Movement-Controlled Tetris Game (EOG Tetris)

An innovative Human-Computer Interaction (HCI) approach leveraging Electrooculography (EOG) technology to allow users to play Tetris using eye movements.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Limitations & Future Work](#limitations--future-work)

## Introduction

EOG Tetris provides an innovative HCI interface that enables users to play the classic Tetris game using their eye movements. This technology has significant implications, especially for users with disabilities, providing them an accessible means to engage with technology.

## Features

- **Signal Processing:** Efficient processing of EOG signals to detect eye movements accurately.
- **Model Selection:** Implements multiple machine learning models, including Decision Trees, SVM, KNN, and Random Forest, with optimized parameters.
- **Live Testing:** Real-time testing capabilities to validate model accuracy and responsiveness.
- **Low Latency:** Dedicated optimizations to ensure minimal delay between eye movement and in-game actions.
- **Configurable Electrode Placement:** Flexibility in electrode placement to ensure accurate blink signal detection.
- **Interactive User Interface:** An engaging Tetris game interface modified for eye movement controls.

## Setup & Installation

1. **Hardware Requirements:** SpikerBox or equivalent EOG hardware.
2. **Software Requirements:** Python3, Arduino SDK, required Python packages (`numpy`, `pandas`, `scikit-learn`, `tsfresh`, `pyautogui`).

## Usage

1. Connect your EOG hardware to your system.
2. Run the game script:
   ```bash
   python tetris_game.py
   ```
3. Adjust your electrode placement as needed (refer to the provided guide on placements).
4. Play Tetris using your eye movements!

## Results

The system demonstrates competitive accuracy levels with:
- Left Movements: 85%
- Right Movements: 80%
- Blink: 80%

For more detailed results, including model accuracies and live test evaluations, refer to our [comprehensive project report](#).

## Limitations & Future Work

- The need for a larger and more representative dataset for improved model training.
- Potential for latency reductions by further optimization.
- Exploring advanced filtering techniques for better signal processing.
- Reevaluating our event detection strategy for enhanced accuracy.

For a detailed discussion on limitations and potential future work, see the [project report](#).

# Snake Game with NEAT AI

An implementation of the classic Snake game with AI trained using NeuroEvolution of Augmenting Topologies (NEAT).

## Overview

Three gameplay modes:
- Human-controlled
- AI training
- AI demonstration

## Setup

```
pip install -r requirements.txt
python main.py
```

## Current Top Score: 36
![Ai playing the game](watchAI.gif)

## Core Components

### Neural Network Architecture
- **Input (28 neurons)**: 24 for 5×5 grid obstacle detection, 4 for food direction
- **Hidden**: Evolves dynamically (initially 18 neurons)
- **Output (4 neurons)**: Directional movement decisions

## Key Algorithms
- **Vision System**: Converts local 5×5 grid and food position into neural inputs
- **Fitness Evaluation**: Each genome plays 3 games, scores averaged
- **Species Formation**: Genomes clustered by structural similarity

## Training Process
1. Initialize population with full input-output connectivity
2. Evaluate performance across multiple games
3. Group similar genomes into species
4. Select top performers for reproduction with mutations
5. Continue until fitness threshold (100) or 50 generations

## File Structure

- `main.py`: Game implementation and NEAT integration
- `config-feedforward.txt`: Neural network evolution parameters
- `best_genome.pkl` & `best_config.pkl`: Trained AI model storage

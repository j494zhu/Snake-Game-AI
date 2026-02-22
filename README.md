# Onyx Snake AI: Vectorized RL Environment & Dueling DQN Architecture

## 🌟 Project Overview
This project implements a state-of-the-art, high-performance Reinforcement Learning (RL) ecosystem for the classic Snake game. It features a custom-built, massively parallel game engine written in Rust and a sophisticated Hybrid Dueling Double DQN trained in PyTorch.

The core infrastructure bypasses the Python Global Interpreter Lock (GIL) via PyO3, allowing 128 game environments to run concurrently. This dramatically accelerates data collection, enabling deep RL models to learn complex pathfinding and obstacle avoidance strategies within hours.

## 🧠 Development Philosophy: AI-Assisted Engineering
The core game engine's memory management, data structures, and reward shaping were designed using a rigorous C++ engineering mindset. I then leveraged LLMs (Claude 3 Opus) as intelligent compilers to accelerate the translation of these designs into memory-safe, concurrent Rust code. This modern AI-assisted workflow allowed me to focus purely on system architecture, mathematical optimization, and RL algorithm tuning.

## 🚀 Core Contributions & System Architecture

### 1. High-Performance RL Engine (Rust & Rayon)
* **Algorithmic Data Structures:** Engineered the core game logic using `VecDeque` for O(1) snake body updates and flattened 1D bitmaps for O(1) collision detection.
* **Reachability Analysis (Flood Fill):** Implemented a custom Breadth-First Search (BFS) algorithm to calculate the percentage of reachable safe space. This provides a crucial dense reward signal, granting the AI the "intuition" to avoid dead ends.
* **Vectorized Execution:** Built a parallel environment manager using `Rayon` to run multiple game instances concurrently across CPU cores, maximizing RL training throughput.

### 2. State-of-the-Art Deep Reinforcement Learning (PyTorch)
* **Hybrid Dueling Double DQN:** Architected a neural network that decouples value and advantage streams. It utilizes a Convolutional Neural Network (CNN) to process the 3x20x20 spatial grid and a Multi-Layer Perceptron (MLP) to process 21-dimensional auxiliary features (distances, lengths, collision warnings).
* **Double Q-Learning Integration:** Decoupled action selection from value estimation during the training step to mitigate Q-value overestimation.
* **Custom Experience Replay:**  Implemented a highly optimized, pre-allocated Numpy replay buffer (capacity: 1,000,000 transitions) to eliminate Python memory allocation overhead during batch sampling.
* **Advanced Training Stability:** Integrated Soft Target Updates ($\tau=0.005$), Cosine Annealing Learning Rate scheduling, Huber Loss (Smooth L1), and Gradient Clipping to ensure steady convergence over extended 9-hour training sessions.

### 3. Cross-Language Integration & Deployment (PyO3)
* **Zero-Overhead Memory Transfer:** Designed batched tensor return mechanisms utilizing PyO3 and Numpy arrays to transfer complex 3D and 2D state representations directly from Rust to Python memory space.
* **Production Inference API:** Decoupled the heavy training loop from the production environment, deploying a lightweight, memory-efficient Flask REST API (`torch.no_grad()`, `model.eval()`) to serve the trained model.

## 🛠️ Tech Stack
* **Systems & Parallelism:** Rust, Cargo, Rayon, PyO3
* **Deep Learning:** Python, PyTorch, Numpy, Reinforcement Learning (Dueling Double DQN)
* **Deployment & Networking:** Flask, Gunicorn, DigitalOcean, Nginx
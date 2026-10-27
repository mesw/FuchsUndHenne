# Fuchs und Henne (Fox and Geese) - Asymmetric Deep RL & WebAssembly Deployment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24%2B-brightgreen.svg)](https://pettingzoo.farama.org/)
[![Qt 6.8.3](https://img.shields.io/badge/Qt-6.8.3%20Quick%20%2F%20QML-41CD52.svg)](https://www.qt.io/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Single--Threaded-654FF0.svg)](https://webassembly.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

An end-to-end deep reinforcement learning (RL) project solving the asymmetric European zero-sum board game **Fuchs und Henne** (Fox and Geese). Features GPU-vectorized PPO self-play, league checkpointing, an interactive Pygame spectator, an optimized 64-bit C++ Bitboard engine, and a responsive **Qt 6.8.3 Quick (QML) WebAssembly** deployment supporting desktop and mobile devices down to 320px width.

> **Live WebAssembly Demo:** [mesw.github.io/FuchsUndHenne](https://mesw.github.io/FuchsUndHenne/)

---

## 1. Game Overview & Rules (Traditional Bavarian/German Rules)

The game is played on a **33-point cross board** embedded within a $7 \times 7$ grid where the four $2 \times 2$ corners are eliminated. Every $2 \times 2$ cell features a full diagonal lattice, forming an Alquerque-style connection graph.

| Row | Col 0 | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 |
| :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | — | — | **0** *(Stall)* | **1** *(Stall)* | **2** *(Stall)* | — | — |
| **1** | — | — | **3** *(Stall)* | **4** *(Stall)* | **5** *(Stall)* | — | — |
| **2** | **6** | **7** | **8** 🦊 *(Fox 1 / Stall)* | **9** *(Stall)* | **10** 🦊 *(Fox 2 / Stall)* | **11** | **12** |
| **3** | **13** 🐔 | **14** 🐔 | **15** 🐔 | **16** 🐔 | **17** 🐔 | **18** 🐔 | **19** 🐔 |
| **4** | **20** 🐔 | **21** 🐔 | **22** 🐔 | **23** 🐔 | **24** 🐔 | **25** 🐔 | **26** 🐔 |
| **5** | — | — | **27** 🐔 | **28** 🐔 | **29** 🐔 | — | — |
| **6** | — | — | **30** 🐔 | **31** 🐔 | **32** 🐔 | — | — |

* **Legend:** 🦊 = Starting Fox position | 🐔 = Starting Hen position | *(Stall)* = 9-node goal square (Rows 0–2, Cols 2–4) | `—` = Cutout corner (invalid)

### Pieces & Placement
* **2 Foxes (Crimson):** Start at the bottom corners of the Stall: `(2,2)` (Node 8) and `(2,4)` (Node 10).
* **20 Hens (Gold):** Start on rows 3, 4, 5, and 6 on the opposite side (Nodes 13 to 32; $7+7+3+3=20$).
* **The Stall ("Der Stall"):** The $3 \times 3$ square of 9 fields at rows 0, 1, 2 and cols 2, 3, 4.

### Mechanics
* **Hens (Move First):** Can move 1 step along valid lines **Forward or Sideways** (Up, Left, Right, Up-Left, Up-Right). Hens cannot move backward and cannot jump.
* **Foxes:** Can move 1 step in **any direction** (orthogonal or diagonal, 8 directions). Alternatively, a Fox can execute a **capture jump** over an adjacent Hen into an empty landing node directly behind it along the line. Multi-jumps are permitted.
* **3-Leerzüge Rule:** Hens are not allowed to oscillate endlessly back and forth between identical positions (maximum 3 consecutive empty/oscillating moves before a forward move is mandated; if no forward move is possible, Hens lose).

### Terminal Conditions
* **Hen Victory ($+1$):**
  1. **Stall Filled:** All 9 Stall nodes are occupied by Hens.
  2. **Foxes Trapped:** Both Foxes have 0 legal moves remaining.
* **Fox Victory ($-1$):**
  1. **Hens Depleted:** Hen count drops to $\le 8$ (12 Hens captured; Hens can no longer fill the Stall).
  2. **Hens Trapped:** Surviving Hens have 0 legal moves remaining.
* **Draw ($0$):** Maximum turn limit reached without decisive victory.

---

## 2. Machine Learning Architecture

* **Observation Space ($4 \times 7 \times 7$ Tensor):**
  * Channel 0: Fox binary mask (up to 2 Foxes).
  * Channel 1: Hen binary mask (up to 20 Hens).
  * Channel 2: Static board mask (valid 33 points).
  * Channel 3: Stall mask (9 stall positions).
* **Action Space ($33 \times 16 = 528$ Discrete Actions):**
  * Origin Node Index: $\lfloor a / 16 \rfloor \in [0, 32]$
  * Direction ID: $a \pmod{16} \in [0, 15]$ (8 step directions + 8 jump directions).
* **Logit Masking:** Illegal actions are suppressed during inference and exploration:
  $$\tilde{z}_i = z_i + (M_i - 1.0) \times 10^9$$
* **Neural Network (`ActorCritic`):**
  * Shared Backbone: $\text{Conv2D}(4 \to 32) \to \text{ReLU} \to \text{Conv2D}(32 \to 64) \to \text{Flatten}(3136)$.
  * Actor Head: $\text{Linear}(3136 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 528)$.
  * Critic Head: $\text{Linear}(3136 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 1) \to \tanh$.
* **League Checkpoint Pool:** Prevents cyclic strategy forgetting by sampling opponents with 70% probability from the frontier policy and 30% uniformly from historical snapshots.
* **Vectorized Training (`train_vectorized.py`):** Runs 256 parallel environments simultaneously on GPU, achieving over **1,300 FPS** (steps/second) for high-throughput self-play.

---

## 3. Repository Structure

```text
FuchsUndHenne/
├── fuchs_und_henne/
│   ├── __init__.py
│   ├── board.py              # 33-node board constants, precomputed lookup tables & move rules
│   ├── env.py                # PettingZoo AEC Environment wrapper with action masking & 3-Leerzüge rule
│   ├── models.py             # ActorCritic CNN with logit masking & ExportableActor
│   ├── league.py             # League pool with 70/30 frontier/historical opponent sampler
│   └── ppo.py                # Rollout buffer, GAE advantages & clipped surrogate update
├── tests/
│   ├── test_board.py         # Move generation, jumps, diagonals, and terminal condition unit tests
│   └── test_env.py           # PettingZoo API compliance & rollout tests
├── QtFuchsUndHenne/          # Qt Quick (QML) & WebAssembly Application
│   ├── bitboard.hpp          # 64-bit Bitboard C++ engine (>5M moves/sec)
│   ├── inference.hpp         # Zero-dependency header-only C++ forward pass (<0.3 ms)
│   ├── gameengine.hpp/.cpp   # Qt/C++ Bridge connecting QML to Bitboard & Neural Networks
│   ├── Main.qml              # Responsive dark-theme QML UI (desktop & mobile down to 320px)
│   ├── test_bitboard.cpp     # C++ engine unit test suite & speed benchmarks
│   ├── CMakeLists.txt        # MinGW Desktop & Emscripten WebAssembly CMake build configuration
│   ├── fox_actor.bin         # Exported Float32 neural network binary weights for Fox
│   └── hens_actor.bin        # Exported Float32 neural network binary weights for Hens
├── exported_models/          # Exported ONNX graphs and C++ binary weights
├── spectator.py              # Interactive 60 FPS Pygame GUI spectator & playtest mode
├── train_vectorized.py       # High-throughput GPU vectorized self-play training loop (256 envs)
├── train.py                  # Standard PPO multi-agent self-play training loop
├── export_onnx.py            # ONNX & binary weight exporter with numerical validation
├── publish-gh-pages.ps1      # Automated GitHub Pages publisher for WebAssembly release builds
├── index.html                # Root landing & redirect page for GitHub Pages
├── requirements.txt          # Pinned Python package dependencies
├── pyproject.toml            # Package & pytest configuration
├── pytest.ini                # Pytest configuration
└── .gitignore                # Comprehensive Git ignore rules (builds, caches, checkpoints)
```

---

## 4. Getting Started

### 4.1 Environment Setup

1. **Create and activate a Python virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   * With NVIDIA GPU (CUDA):
     ```powershell
     pip install torch --index-url https://download.pytorch.org/whl/cu124
     pip install -r requirements.txt
     ```
   * CPU only:
     ```powershell
     pip install -r requirements.txt
     ```

### 4.2 Run Unit Tests

Execute the comprehensive test suite (all 17 tests):
```powershell
pytest -v
```

### 4.3 Launch Self-Play Training

Train the Fox and Hen policies via vectorized self-play on GPU:
```powershell
# Vectorized high-speed training (256 parallel envs):
python train_vectorized.py --total_steps 500000 --num_envs 256 --buffer_size 4096

# Resume from existing weights:
python train_vectorized.py --resume --total_steps 500000 --num_envs 256 --buffer_size 4096
```

Or run standard sequential training:
```powershell
python train.py --total_steps 100000 --eval_interval 5
```

### 4.4 Monitor with TensorBoard

Start TensorBoard in a separate terminal:
```powershell
tensorboard --logdir runs --reload_multifile true
```
Open `http://localhost:6006` to inspect loss curves, win rates, entropy, and episode lengths.

---

## 5. Interactive Pygame Spectator & Playtesting

Run the standalone visual spectator to watch trained models or play directly against the AI:

```powershell
# Run with unlimited turns:
python spectator.py --max_turns 0

# Or run with a custom turn limit:
python spectator.py --max_turns 300
```

### Spectator Hotkeys & Controls

| Key / Input | Action |
| :--- | :--- |
| **Mouse Click** | Click a piece to highlight legal destinations; click an indicator to move |
| **`SPACE`** | Pause / Resume AI autoplay |
| **`M`** | Cycle Game Mode (*AI vs AI* $\to$ *Human as Hens* $\to$ *Human as Fox*) |
| **`T`** | Toggle Style (*Stochastic / Varied Play* $\leftrightarrow$ *Greedy / Argmax*) |
| **`L`** | Hot-reload the latest checkpoints from disk while training runs |
| **`R`** | Reset board to starting positions |
| **`UP` / `DOWN`** | Adjust move speed (1 to 30 moves/sec) |

---

## 6. ONNX Export & C++ Bitboard Inference Engine

### 6.1 Model Export & Numerical Validation
The trained PyTorch policies are exported using `export_onnx.py`. This script extracts the Actor head, integrates legal action masking directly into the computation graph, validates numerical parity with ONNX Runtime ($0.0$ error), produces lightweight binary weights, and automatically synchronizes them to the Qt build folders:

```powershell
python export_onnx.py --fox fox_final.pt --hens hens_final.pt
```

Outputs:
* `exported_models/fox_actor.onnx` & `exported_models/hens_actor.onnx`
* `exported_models/fox_actor.bin` & `exported_models/hens_actor.bin` (auto-synced to `QtFuchsUndHenne/` and build directories)

### 6.2 C++ Bitboard Engine & Inference Performance
* **64-bit Bitboard Representation (`bitboard.hpp`):** Precomputed step/jump tables, full diagonal lattice geometry, and jump-chaining. Generates $> 5{,}000{,}000$ legal actions/second ($< 100\text{ ns}$ execution).
* **Pure C++ Zero-Dependency Forward Pass (`inference.hpp`):** Executes Conv2D + Linear layers with embedded action masking in under $300\,\mu\text{s}$ per inference, eliminating heavyweight runtime dependencies for single-threaded WebAssembly.
* **C++ Unit Tests & Benchmarks:**
  ```powershell
  .\QtFuchsUndHenne\build\Desktop_Qt_6_8_3_MinGW_64_bit-Debug\test_bitboard.exe
  ```

---

## 7. Qt Quick (QML) 6.8.3 & WebAssembly Deployment

The application features a sleek dark-theme user interface built with **Qt Quick (QML)** that compiles natively for Desktop (MinGW) and WebAssembly (Emscripten).

### 7.1 Responsive Design (Desktop to 320px Mobile)
* **Desktop Landscape (`width >= 720`):** 2-column layout with the board centered on the left and a fixed 340px control sidebar on the right.
* **Mobile Portrait (`width < 720`):** 1-column layout with the board proportionally scaled at the top, and cards stacked vertically below.
* **Minimum Width 320px Support:** Engineered and verified for small mobile viewports (e.g. iPhone SE at 320×568) with touch-friendly tap targets, native-style smooth flick scrolling, and a floating scroll position indicator.
* **AI Play Style (Varied vs. Greedy):** Supports both **Varied Play** (Softmax temperature sampling with $\tau = 0.7$ over legal moves for diverse and exciting AI vs AI games) and **Greedy Play** (pure argmax deterministic play), switchable via UI button or the `T` key.

### 7.2 Building WebAssembly Locally
Ensure Emscripten is configured and run:
```powershell
$env:EMSDK = "C:\path\to\emsdk"
$env:PATH = "C:\Qt\Tools\CMake_64\bin;C:\Qt\Tools\Ninja;C:\Qt\6.8.3\wasm_singlethread\bin;" + $env:PATH
cmake --build QtFuchsUndHenne/build/WebAssembly_Qt_6_8_3_single_threaded-Release
```
The build automatically embeds `fox_actor.bin` and `hens_actor.bin` via `--preload-file` into `appQtFuchsUndHenne.data` for zero-latency client-side inference.

To test locally:
```powershell
python -m http.server 8088 --directory QtFuchsUndHenne/build/WebAssembly_Qt_6_8_3_single_threaded-Release
```
Open `http://localhost:8088/appQtFuchsUndHenne.html`.

### 7.3 Publishing to GitHub Pages
To publish the latest WebAssembly build to your GitHub Pages `gh-pages` branch:
```powershell
.\publish-gh-pages.ps1
```

---

## 8. Implementation Roadmap

- [x] **Step 1: PPO Self-Play Training Pipeline**
  - [x] 33-point cross board with full diagonal lattice & precomputed lookups
  - [x] PettingZoo AEC Environment with move masking & multi-jumps
  - [x] League checkpoint buffer & 70/30 opponent sampler
  - [x] GAE + clipped PPO loss + TensorBoard telemetry
  - [x] GPU-accelerated vectorized training engine (256 parallel environments, >1,300 FPS)
- [x] **Step 2: Pygame GUI Spectator Window**
  - [x] 60 FPS interactive graphical board
  - [x] Human vs. AI playtest mode (Play as Fox / Play as Hens)
  - [x] 3-Leerzüge repetition breaker & dynamic critic win evaluation
- [x] **Step 3: ONNX Export & C++ Bitboard Inference Engine**
  - [x] Actor model export to ONNX graph with embedded masking (`export_onnx.py`)
  - [x] Numerical validation against ONNX Runtime (0.0 error margin)
  - [x] 64-bit Bitboard C++ move generator & state engine (>5M moves/sec, <100 ns execution)
  - [x] Header-only zero-dependency C++ forward pass (<0.3 ms latency)
  - [x] Standalone test suite & benchmark (`test_bitboard.cpp`)
- [x] **Step 4: Qt Quick (QML) 6.8.3 WebAssembly Deployment**
  - [x] Single-threaded Emscripten compilation with preloaded neural network weights (`.data`)
  - [x] Responsive touch/mouse QML interface (Desktop 2-column + Mobile 1-column layout)
  - [x] Minimum 320px screen width support with smooth scrolling and touch targets
  - [x] One-click GitHub Pages publisher (`publish-gh-pages.ps1`) & landing page (`index.html`)

---

## 9. License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](LICENSE).

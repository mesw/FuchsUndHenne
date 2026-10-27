# Project Specification: Asymmetric Deep RL & WASM Deployment (Fuchs und Henne)

## 1. Project Overview & Objectives

This project implements an end-to-end deep reinforcement learning (RL) pipeline for the traditional European board game **Fuchs und Henne** (Fox and Geese). The primary engineering and research objectives are:

1. **Solve Asymmetric Zero-Sum Dynamics:** Train two structurally distinct adversarial policies—a predatory, agile agent (**Fox**) and a coordinated territorial swarm agent (**Hens**)—using local multi-agent self-play.
2. **Prevent Strategy Cycling:** Implement a league-style checkpoint pool to drive convergence toward approximate Nash equilibrium without circular policy forgetting.
3. **Inspect Strategy Evolution Visually:** Monitor training through real-time telemetry (TensorBoard) and a live visual spectator (Pygame).
4. **Deploy to Single-Threaded WebAssembly:** Export the trained actors to a compact format and run real-time inference ($<0.5\text{ ms}$) inside a client-side C++ / Qt 6.8.3 WebAssembly application hosted on GitHub Pages.

---

## 2. Game Rules & Mathematical Formalism (Traditional German Rules)

### 2.1 The Board Topology & Diagonal Lattice
The game takes place on a traditional **33-point cross board** embedded within a $7 \times 7$ grid where the four $2 \times 2$ corners are eliminated. Every $2 \times 2$ cell on the board contains full diagonal connections, forming an Alquerque-style lattice:

```
       (0,2)───(0,3)───(0,4)
       │  ╲   ╱ │ ╲   ╱  │
       │   ╲ ╱  │  ╲ ╱   │
       (1,2)───(1,3)───(1,4)
       │  ╱ ╲   │ ╱ ╲    │
       │ ╱   ╲  │╱   ╲   │
 (2,0)─(2,1)───(2,2)───(2,3)───(2,4)───(2,5)─(2,6)
 │ ╲ ╱ │ ╲ ╱   │ ╲ ╱   │ ╲ ╱   │ ╲ ╱   │ ╲ ╱ │
 (3,0)─(3,1)───(3,2)───(3,3)───(3,4)───(3,5)─(3,6)
 │ ╱ ╲ │ ╱ ╲   │ ╱ ╲   │ ╱ ╲   │ ╱ ╲   │ ╱ ╲ │
 (4,0)─(4,1)───(4,2)───(4,3)───(4,4)───(4,5)─(4,6)
       │ ╲   ╱ │ ╲   ╱ │
       (5,2)───(5,3)───(5,4)
       │ ╱   ╲ │ ╱   ╲ │
       (6,2)───(6,3)───(6,4)
```

The board contains $|V| = 33$ nodes, indexed $0$ through $32$ in row-major order:
* **Stall ("Der Stall"):** The $3 \times 3$ square of 9 fields at rows 0, 1, 2 and cols 2, 3, 4 (Nodes 0, 1, 2, 3, 4, 5, 8, 9, 10).

### 2.2 Asymmetric Roles & Mechanics
* **Pieces & Initial Placement:**
  * **Foxes ($F$):** 2 pieces, starting at the lower corner nodes of the Stall: `(2, 2)` (Node 8) and `(2, 4)` (Node 10).
  * **Hens ($H$):** 20 pieces, occupying all valid positions on the opposite side across rows 3, 4, 5, and 6 (Nodes 13 to 32; $7 + 7 + 3 + 3 = 20$).
* **Move Mechanics:**
  * **Hens (Move First):** Can move 1 step along valid orthogonal or diagonal lines **Forward or Sideways**:
    * Allowed directions: Up `(-1, 0)`, Left `(0, -1)`, Right `(0, +1)`, Up-Left `(-1, -1)`, Up-Right `(-1, +1)`.
    * Hens cannot move backward (Down, Down-Left, Down-Right) and cannot jump.
  * **Foxes:** Can move 1 step in **any** valid direction (orthogonal or diagonal, 8 directions).
    * Alternatively, a Fox can execute a **capture jump** over an adjacent Hen into an empty landing node directly behind it along the line (orthogonal or diagonal). Multi-jumps are permitted. Captured Hens are removed.
* **Terminal Conditions:**
  * **Hen Victory ($R_{\text{hen}} = +1, R_{\text{fox}} = -1$):**
    1. **Stall Filled:** All 9 nodes of the Stall are occupied by Hens.
    2. **Foxes Trapped:** Both Foxes have 0 legal moves remaining.
  * **Fox Victory ($R_{\text{fox}} = +1, R_{\text{hen}} = -1$):**
    * Hen count drops to $\le 8$ (meaning 12 Hens are captured; Hens can no longer fill the 9-node Stall).
  * **Draw / Truncation ($R = 0$):** Move count reaches $T_{\max} = 150$ without a decisive win.

---

## 3. Machine Learning Specifications

### 3.1 Observation Space
The state is represented as a tensor $S \in \mathbb{R}^{4 \times 7 \times 7}$:
* **Channel 0:** Binary mask of Fox locations (1 where Fox is present, else 0; up to 2 Foxes).
* **Channel 1:** Binary mask of Hen locations (1 where Hen is present, else 0; up to 20 Hens).
* **Channel 2:** Static board mask (1 for valid 33 cross-board nodes, 0 for eliminated corners).
* **Channel 3:** Stall mask (1 for the 9 Stall nodes, 0 elsewhere).

### 3.2 Action Space & Masking
* **Discrete Action Dimension:** $33 \times 16 = 528$ actions.
* An action ID $a \in [0, 527]$ maps to:
  $$\text{Origin Node Index} = \lfloor a / 16 \rfloor, \quad \text{Direction ID} = a \pmod{16}$$
* **Directions:**
  * **Steps (0–7):**
    * `0: Up (-1, 0)`, `1: Down (+1, 0)`, `2: Left (0, -1)`, `3: Right (0, +1)`
    * `4: Up-Left (-1, -1)`, `5: Up-Right (-1, +1)`, `6: Down-Left (+1, -1)`, `7: Down-Right (+1, +1)`
  * **Jumps (8–15):**
    * `8: Jump-Up (-2, 0)`, `9: Jump-Down (+2, 0)`, `10: Jump-Left (0, -2)`, `11: Jump-Right (0, +2)`
    * `12: Jump-Up-Left (-2, -2)`, `13: Jump-Up-Right (-2, +2)`, `14: Jump-Down-Left (+2, -2)`, `15: Jump-Down-Right (+2, +2)`
* **Invalid Action Masking:** Actor logits $z \in \mathbb{R}^{528}$ are modified using binary mask $M \in \{0, 1\}^{528}$:
  $$\tilde{z}_i = z_i + (M_i - 1.0) \times 10^9$$
  $$\pi(a_i \mid s) = \frac{\exp(\tilde{z}_i)}{\sum_j \exp(\tilde{z}_j)}$$

### 3.3 Network Architecture
* **Feature Extractor:**
  * $\text{Conv2D}(4 \to 32, \text{kernel}=3, \text{padding}=1) \to \text{ReLU}$
  * $\text{Conv2D}(32 \to 64, \text{kernel}=3, \text{padding}=1) \to \text{ReLU}$
  * $\text{Flatten} \to 64 \times 7 \times 7 = 3136\text{ units}$
* **Actor Head:** $\text{Linear}(3136 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 528)$
* **Critic Head:** $\text{Linear}(3136 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 1) \to \tanh$

---

## 4. Implementation Roadmap: Next Steps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PPO Self-Play Training Pipeline                                      │
│ • PettingZoo AEC Environment Loop                                           │
│ • League Checkpoint Buffer (Past vs. Current)                               │
│ • GAE + Masked PPO Loss Optimization + TensorBoard Logging                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Pygame GUI Spectator Window                                         │
│ • Standalone 60 FPS visual evaluation harness                               │
│ • Interpolated piece animation & real-time decision inspection               │
│ • Manual Human vs. AI playtest mode                                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: ONNX Export & C++ Inference Engine                                  │
│ • PyTorch -> ONNX Actor Graph Tracing                                       │
│ • C++ 64-bit Bitboard representation & legal move engine                    │
│ • Sub-millisecond tensor forward pass evaluation                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Qt Quick 6.8.3 WebAssembly Deployment                               │
│ • Emscripten compilation pipeline (single-threaded target)                  │
│ • Responsive QML UI (Touch + Mouse controls, board animations)              │
│ • GitHub Actions CI/CD to GitHub Pages static host                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Full PPO Self-Play Training Pipeline

#### 1.1 League Checkpoint Management
To counteract policy cycling (where the Fox optimizes for a temporary Hen vulnerability, and the Hens abandon solid positional containment to patch that single leak):
* Maintain two separate model pools: `FoxLeague` and `HenLeague`.
* At the start of each rollout batch, the active Fox agent selects an opponent from `HenLeague` using a biased distribution:
  * $70\%$ probability: Most recent Hen checkpoint (driving the frontier of play).
  * $30\%$ probability: Uniformly sampled historical Hen checkpoint (enforcing robust baselines).
* Every $K$ updates (e.g., 200 epochs), save a frozen copy of both policies into their respective league pools.

#### 1.2 Generalized Advantage Estimation (GAE) & Clipping
* Experience tuples store `(obs, mask, action, log_prob, reward, value, done)`.
* Compute advantages $\hat{A}_t$ via GAE:
  $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
* Optimize the standard clipped surrogate objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  with entropy bonus $S[\pi_\theta]$ to maintain exploration in high-branching states.

#### 1.3 Telemetry & Logging
Log the following metrics to TensorBoard (`runs/fuchs_und_henne_v0`):
* `win_rate/fox` vs. `win_rate/hens` (evaluated over 100 greedy matches every 50 epochs).
* `game/mean_episode_length` (tracking defense depth; games should lengthen as Hens learn to wall).
* `policy/entropy` (ensuring steady decay without premature collapse).
* `losses/actor_loss`, `losses/critic_loss`, and `losses/explained_variance`.

---

### Step 2: Pygame GUI Spectator Window

#### 2.1 Functionality
The spectator provides a local graphical window allowing the developer to:
1. **Watch Training Checkpoints:** Render non-deterministic or greedy policy matches at adjustable speeds ($1\text{–}60\text{ FPS}$).
2. **Interactive Play:** Let a human player control either the Fox or the Hens against the latest checkpoint to test exploitability.
3. **Visual Heuristics:** Display an overlay of the policy distribution (arrow indicators with alpha proportional to move probability) and the critic's current win-probability assessment.

#### 2.2 Board Rendering Architecture
* **Coordinate System:** Map valid $(r, c)$ board indices to pixel coordinates:
  $$x = \text{OFFSET}_X + c \times \text{CELL\_SIZE}, \quad y = \text{OFFSET}_Y + r \times \text{CELL\_SIZE}$$
* **Visual States:**
  * Background: Board cross outline with valid connection lines.
  * Hens: Amber/White circular tokens with smooth positional lerping.
  * Fox: Crimson circular token with highlight rings indicating jump range.
  * Status Bar: Turn count, active agent, Hen casualty counter, and evaluation score bar.

---

### Step 3: ONNX Export & C++ Inference Engine

#### 3.1 Model Export Strategy
Only the **Actor head** is required for deployment; the Critic head can be stripped:
```python
class ExportableActor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv = model.conv
        self.actor_head = model.actor_head

    def forward(self, obs, mask):
        feat = self.conv(obs)
        logits = self.actor_head(feat)
        # Apply mask directly inside ONNX graph
        masked_logits = logits + (mask - 1.0) * 1e9
        return torch.softmax(masked_logits, dim=-1)
```
Export using `torch.onnx.export` with inputs:
* `obs`: Float32 tensor of shape `(1, 4, 7, 7)`
* `mask`: Float32 tensor of shape `(1, 528)`
* Output: Action probability vector of shape `(1, 528)`

#### 3.2 High-Performance C++ Bitboard Representation
To maximize speed in single-threaded WebAssembly, replace matrix arrays with **64-bit bitboards**:
```cpp
struct BitBoard {
    uint64_t fox;      // Exactly 1 bit set among the 33 valid bits
    uint64_t hens;     // Up to 15 bits set among the 33 valid bits
    bool fox_turn;
};
```
* **Bitboard Operations:** Move generation, collision detection, and jump validations execute via bitwise shifts and precomputed lookup masks (`ADJACENT_ORTHO[33]`, `JUMP_OVER[33][4]`, `JUMP_LAND[33][4]`).
* Execution cost per move validation: $<100\text{ nanoseconds}$.

#### 3.3 Inference Runtime Choices in C++
* **Approach A: ONNX Runtime (WASM):** Compile ONNX Runtime Web/C++ via Emscripten (`ort-wasm.wasm`). Reliable, supports operator fusion.
* **Approach B: Header-Only Matrix Pass (Zero Dependency):** Since the model consists of standard Conv2D and Linear layers, an optimized header-only tensor implementation (e.g., via Eigen or minimal matrix multiplication routines) compiles directly into the Qt binary, avoiding asynchronous `.wasm` asset loading issues.

---

### Step 4: Qt Quick (QML) & Qt 6.8.3 WebAssembly Game

#### 4.1 UI & UX Design (Qt Quick / QML)
* **Declarative Scene:** Built entirely in QML using `QtQuick 2.15` and `QtQuick.Controls 2.15`.
* **State Machine:**
  * `MENU_SCREEN`: Select sides (Play as Fox, Play as Hens, or Watch AI vs. AI), select AI difficulty checkpoint.
  * `GAMEPLAY_SCREEN`: Interactive 33-point board with touch/click responsiveness and drag-and-drop or tap-to-move input.
  * `GAME_OVER_SCREEN`: Victory banner with restart options and move recap.
* **Visual Polish:** QML `NumberAnimation` and `SpringAnimation` for smooth piece translations and capture fades.

#### 4.2 C++ / QML Bridge (`GameController`)
A central C++ `QObject` exposed to QML:
```cpp
class GameEngine : public QObject {
    Q_OBJECT
    Q_PROPERTY(int activePlayer READ activePlayer NOTIFY stateChanged)
    Q_PROPERTY(int hensRemaining READ hensRemaining NOTIFY stateChanged)
    Q_PROPERTY(bool isThinking READ isThinking NOTIFY stateChanged)

public slots:
    void handleUserMove(int sourceNode, int targetNode);
    void triggerAIMove();
    void resetGame();

signals:
    void stateChanged();
    void moveExecuted(int src, int dst, bool isCapture, int capturedNode);
    void gameOver(int winner);
};
```

#### 4.3 WebAssembly Compilation & Single-Thread Configuration
Targeting Qt 6.8.3 with Emscripten:
* **Single-Threaded Execution:** Built without `pthreads` to avoid SharedArrayBuffer restrictions, allowing deployment on standard GitHub Pages without requiring custom Cross-Origin Isolation headers (`COOP`/`COEP`).
* **Compiler Flags:**
  ```cmake
  target_link_options(FuchsUndHenne PRIVATE
      "-sWASMSE_ASYNC_COMPILATION=1"
      "-sALLOW_MEMORY_GROWTH=1"
      "-sINITIAL_MEMORY=64MB"
      "-sMAXIMUM_MEMORY=256MB"
      "-O3"
  )
  ```
* **Asset Bundling:** Game icons, board graphics, and the model weights (`fox_actor.onnx`, `hens_actor.onnx`) are embedded directly into the binary via Qt's Resource System (`qrc:/`).

#### 4.4 Continuous Deployment to GitHub Pages
A GitHub Actions workflow:
1. Sets up the Emscripten SDK matching the Qt 6.8.3 WebAssembly toolchain.
2. Builds the project using `cmake --build . --target FuchsUndHenne`.
3. Packages `index.html`, `FuchsUndHenne.js`, `FuchsUndHenne.wasm`, and `qtloader.js`.
4. Deploys the static artifacts to the `gh-pages` branch.

---

## 5. Development Milestones & Checklist

| Milestone | Deliverables | Verification Criteria |
| :--- | :--- | :--- |
| **M1: Core Environment & Baseline** | Complete PettingZoo AEC environment with move masking and test scripts | Unit tests pass for move generation, jump captures, and illegal move filtering |
| **M2: Self-Play Training Loop** | PPO implementation with league checkpoint archive and TensorBoard logging | Hens learn basic blocking walls; average episode length exceeds 40 turns |
| **M3: Pygame Visual Spectator** | GUI script rendering board state, live move execution, and interactive human mode | Smooth 60 FPS playback; user can play against exported PyTorch weights |
| **M4: ONNX Export & C++ Core** | Stripped Actor export script + C++ bitboard validation suite | C++ move generator matches Python state transitions $100\%$ over $10^5$ random games |
| **M5: Qt 6.8.3 WASM Client** | QML board interface + C++ inference wrapper + Emscripten CMake configuration | Game runs in browser at 60 FPS with zero dropped frames and $<1\text{ ms}$ AI latency |
| **M6: GitHub Pages Deployment** | GitHub Actions workflow for automated building and publishing | Public URL loads and executes client-side AI without server communication |
#ifndef FUCHS_UND_HENNE_INFERENCE_HPP
#define FUCHS_UND_HENNE_INFERENCE_HPP

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "bitboard.hpp"

namespace fuchs {

using ObsTensor = std::array<std::array<std::array<float, GRID_SIZE>, GRID_SIZE>, 4>;

inline ObsTensor make_observation_tensor(const BitBoard& board) {
    ObsTensor obs{};

    for (int u = 0; u < NUM_NODES; ++u) {
        int r = NODE_COORDS[u].r;
        int c = NODE_COORDS[u].c;

        if ((board.foxes >> u) & 1ULL) obs[0][r][c] = 1.0f;
        if ((board.hens >> u) & 1ULL)  obs[1][r][c] = 1.0f;
        obs[2][r][c] = 1.0f; // Valid board node
        if ((STALL_NODES_MASK >> u) & 1ULL) obs[3][r][c] = 1.0f; // Stall node
    }
    return obs;
}

struct ActorWeights {
    std::vector<float> conv1_w; // 32 * 4 * 9 = 1152
    std::vector<float> conv1_b; // 32
    std::vector<float> conv2_w; // 64 * 32 * 9 = 18432
    std::vector<float> conv2_b; // 64
    std::vector<float> fc1_w;   // 256 * 3136 = 802816
    std::vector<float> fc1_b;   // 256
    std::vector<float> fc2_w;   // 528 * 256 = 135168
    std::vector<float> fc2_b;   // 528

    bool is_loaded{false};

    bool load_from_binary(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        auto read_floats = [&](std::vector<float>& vec, size_t count) {
            vec.resize(count);
            file.read(reinterpret_cast<char*>(vec.data()), count * sizeof(float));
        };

        read_floats(conv1_w, 32 * 4 * 3 * 3);
        read_floats(conv1_b, 32);
        read_floats(conv2_w, 64 * 32 * 3 * 3);
        read_floats(conv2_b, 64);
        read_floats(fc1_w, 256 * 3136);
        read_floats(fc1_b, 256);
        read_floats(fc2_w, 528 * 256);
        read_floats(fc2_b, 528);

        is_loaded = file.good();
        return is_loaded;
    }
};

inline int predict_best_action(
    const ActorWeights& w,
    const ObsTensor& obs,
    const std::vector<int>& legal_actions
) {
    if (legal_actions.empty()) return -1;
    if (legal_actions.size() == 1) return legal_actions[0];
    if (!w.is_loaded) return legal_actions[0];

    // Layer 1: Conv2D(4 -> 32, k=3, pad=1) + ReLU
    float c1_out[32][7][7]{};
    for (int oc = 0; oc < 32; ++oc) {
        float bias = w.conv1_b[oc];
        const float* k_base = &w.conv1_w[oc * 36];
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                float sum = bias;
                for (int ic = 0; ic < 4; ++ic) {
                    const float* kw = k_base + ic * 9;
                    for (int dr = -1; dr <= 1; ++dr) {
                        int nr = r + dr;
                        if (nr < 0 || nr >= 7) continue;
                        for (int dc = -1; dc <= 1; ++dc) {
                            int nc = c + dc;
                            if (nc < 0 || nc >= 7) continue;
                            sum += obs[ic][nr][nc] * kw[(dr + 1) * 3 + (dc + 1)];
                        }
                    }
                }
                c1_out[oc][r][c] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    // Layer 2: Conv2D(32 -> 64, k=3, pad=1) + ReLU
    float c2_flat[3136];
    for (int oc = 0; oc < 64; ++oc) {
        float bias = w.conv2_b[oc];
        const float* k_base = &w.conv2_w[oc * 288];
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                float sum = bias;
                for (int ic = 0; ic < 32; ++ic) {
                    const float* kw = k_base + ic * 9;
                    for (int dr = -1; dr <= 1; ++dr) {
                        int nr = r + dr;
                        if (nr < 0 || nr >= 7) continue;
                        for (int dc = -1; dc <= 1; ++dc) {
                            int nc = c + dc;
                            if (nc < 0 || nc >= 7) continue;
                            sum += c1_out[ic][nr][nc] * kw[(dr + 1) * 3 + (dc + 1)];
                        }
                    }
                }
                c2_flat[oc * 49 + r * 7 + c] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    // Layer 3: FC1 Linear(3136 -> 256) + ReLU
    float fc1_out[256];
    for (int i = 0; i < 256; ++i) {
        float sum = w.fc1_b[i];
        const float* w_row = &w.fc1_w[i * 3136];
        for (int j = 0; j < 3136; ++j) {
            sum += c2_flat[j] * w_row[j];
        }
        fc1_out[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    // Layer 4: FC2 Linear(256 -> 528) evaluated ONLY for legal actions!
    int best_act = legal_actions[0];
    float best_logit = -1e9f;

    for (int act : legal_actions) {
        float logit = w.fc2_b[act];
        const float* w_row = &w.fc2_w[act * 256];
        for (int j = 0; j < 256; ++j) {
            logit += fc1_out[j] * w_row[j];
        }
        if (logit > best_logit) {
            best_logit = logit;
            best_act = act;
        }
    }

    return best_act;
}

} // namespace fuchs

#endif // FUCHS_UND_HENNE_INFERENCE_HPP

#include <iostream>
#include <chrono>
#include <cassert>
#include "bitboard.hpp"

using namespace fuchs;

void test_initial_state() {
    BitBoard b;
    b.reset();

    assert(b.fox_count() == 2);
    assert(b.hen_count() == 20);
    assert(!b.fox_turn);
    assert(b.turn_count == 0);

    // Initial legal moves for hens
    auto hen_moves = get_legal_actions(b);
    assert(!hen_moves.empty());
    std::cout << "[+] test_initial_state passed (Initial Hen moves: " << hen_moves.size() << ")\n";
}

void test_diagonal_lattice() {
    const auto& steps = get_step_targets();

    // Node 0 is (0,2). Step in direction 7 (Down-Right) -> Node 4 (1,3)
    assert(steps[0][7] == 4);

    // Node 3 is (1,2). Step in direction 6 (Down-Left) -> (2,1)
    // Missing corner (1,1) -> MUST be -1
    assert(steps[3][6] == -1);

    // Node 5 is (1,4). Step in direction 7 (Down-Right) -> (2,5)
    // Missing corner (1,5) -> MUST be -1
    assert(steps[5][7] == -1);

    std::cout << "[+] test_diagonal_lattice passed (Corner cutouts properly verified)\n";
}

void test_jump_capture() {
    BitBoard b;
    b.foxes = (1ULL << 8); // Fox at node 8: (2,2)
    b.hens = (1ULL << 9);  // Hen at node 9: (2,3)
    b.fox_turn = true;

    // Jump-Right is direction 3 (step Right) + 8 = 11
    int jump_action = encode_action(8, 11);
    auto actions = get_legal_actions(b);

    bool found = false;
    for (int a : actions) {
        if (a == jump_action) { found = true; break; }
    }
    assert(found);

    bool ok = apply_action(b, jump_action);
    assert(ok);
    assert(b.hen_count() == 0); // Hen captured!
    assert((b.foxes >> 10) & 1ULL); // Fox landed on node 10: (2,4)

    std::cout << "[+] test_jump_capture passed (Fox jump capture executed and verified)\n";
}

void test_terminal_conditions() {
    // 1. Stall filled (9 hens in stall -> Hens win +1)
    BitBoard b1;
    b1.hens = STALL_NODES_MASK; // 9 hens occupying stall
    b1.foxes = (1ULL << 13) | (1ULL << 19);
    auto [done1, win1, reason1] = check_game_over(b1);
    assert(done1);
    assert(win1 == 1);
    assert(reason1 == "stall_filled");

    // 2. Hens depleted (<= 8 hens remaining -> Foxes win -1)
    BitBoard b2;
    b2.hens = (1ULL << 13) | (1ULL << 14) | (1ULL << 15) | (1ULL << 16) |
              (1ULL << 17) | (1ULL << 18) | (1ULL << 19) | (1ULL << 20); // 8 hens
    b2.foxes = START_FOXES_MASK;
    auto [done2, win2, reason2] = check_game_over(b2);
    assert(done2);
    assert(win2 == -1);
    assert(reason2 == "hens_depleted");

    // 3. Foxes trapped (Foxes immobilized with 0 moves -> Hens win +1)
    BitBoard b3;
    b3.foxes = (1ULL << 0); // Single fox at node 0: (0, 2)
    // Block steps: 1 (0,3), 3 (1,2), 4 (1,3)
    // Block jump landings: 2 (0,4), 8 (2,2), 10 (2,4)
    b3.hens = (1ULL << 1) | (1ULL << 3) | (1ULL << 4) |
              (1ULL << 2) | (1ULL << 8) | (1ULL << 10) |
              (1ULL << 13) | (1ULL << 14) | (1ULL << 15); // 9 hens (> 8)
    auto [done3, win3, reason3] = check_game_over(b3);
    assert(done3);
    assert(win3 == 1);
    assert(reason3 == "foxes_trapped");

    // 4. Hens trapped (10 Hens blocked with no legal forward/sideways steps -> Foxes win -1)
    BitBoard b4;
    b4.foxes = (1ULL << 8) | (1ULL << 10);
    b4.hens = (1ULL << 0) | (1ULL << 1) | (1ULL << 2) |
              (1ULL << 3) | (1ULL << 4) | (1ULL << 5) |
              (1ULL << 6) | (1ULL << 7) | (1ULL << 11) | (1ULL << 12); // 10 hens
    auto [done4, win4, reason4] = check_game_over(b4);
    assert(done4);
    assert(win4 == -1);
    assert(reason4 == "hens_trapped");

    std::cout << "[+] test_terminal_conditions passed (All 4 victory/loss rules verified)\n";
}

void test_observation_tensor() {
    BitBoard b;
    b.reset();
    auto obs = make_observation_tensor(b);

    // Channel 0: Foxes
    int fox_count = 0;
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            if (obs[0][r][c] > 0.5f) fox_count++;
        }
    }
    assert(fox_count == 2);
    assert(obs[0][2][2] == 1.0f);
    assert(obs[0][2][4] == 1.0f);

    // Channel 1: Hens
    int hen_count = 0;
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            if (obs[1][r][c] > 0.5f) hen_count++;
        }
    }
    assert(hen_count == 20);

    // Channel 2: Board nodes
    int board_nodes = 0;
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            if (obs[2][r][c] > 0.5f) board_nodes++;
        }
    }
    assert(board_nodes == 33);

    // Channel 3: Stall nodes
    int stall_nodes = 0;
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            if (obs[3][r][c] > 0.5f) stall_nodes++;
        }
    }
    assert(stall_nodes == 9);

    std::cout << "[+] test_observation_tensor passed (4x7x7 CNN grid encoding matches specification)\n";
}

void test_inference_and_benchmark() {
    ActorWeights weights;
    const std::string candidate_paths[] = {
        "fox_actor.bin",
        "./fox_actor.bin",
        "../fox_actor.bin",
        "../exported_models/fox_actor.bin",
        "../../exported_models/fox_actor.bin"
    };

    bool loaded = false;
    for (const auto& path : candidate_paths) {
        if (weights.load_from_binary(path)) {
            std::cout << "[+] Neural network weights loaded from: " << path << "\n";
            loaded = true;
            break;
        }
    }

    BitBoard b;
    b.reset();
    b.fox_turn = true;
    auto legal_actions = get_legal_actions(b);
    assert(!legal_actions.empty());
    auto obs = make_observation_tensor(b);

    int best_act = predict_best_action(weights, obs, legal_actions);
    bool valid = false;
    for (int a : legal_actions) {
        if (a == best_act) { valid = true; break; }
    }
    assert(valid);

    if (loaded) {
        constexpr int INFER_ITERS = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < INFER_ITERS; ++i) {
            predict_best_action(weights, obs, legal_actions);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double latency_us = (elapsed_ms * 1000.0) / INFER_ITERS;

        std::cout << "[*] C++ Sparse Forward-Pass Latency: " << latency_us
                  << " us / inference (" << (1000.0 / (latency_us / 1000.0)) << " inferences/sec)\n";
        std::cout << "[+] test_inference_and_benchmark passed (Model evaluated legal action successfully)\n";
    } else {
        std::cout << "[!] Notice: fox_actor.bin not found during runtime search, fallback legal action used\n";
    }
}

void benchmark_move_generation() {
    BitBoard b;
    b.reset();

    constexpr int ITERATIONS = 1'000'000;
    auto start = std::chrono::high_resolution_clock::now();

    size_t total_moves = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto moves = get_legal_actions(b);
        total_moves += moves.size();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    double moves_per_sec = ITERATIONS / elapsed_sec;

    std::cout << "[*] Move Generation Speed: " << (moves_per_sec / 1e6)
              << " MILLION generations / second ("
              << (elapsed_sec * 1e9 / ITERATIONS) << " ns per call)\n";
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "[*] Running Fuchs und Henne C++ Bitboard Test Suite\n";
    std::cout << "=======================================================\n";

    test_initial_state();
    test_diagonal_lattice();
    test_jump_capture();
    test_terminal_conditions();
    test_observation_tensor();
    test_inference_and_benchmark();
    benchmark_move_generation();

    std::cout << "=======================================================\n";
    std::cout << "[+] ALL C++ TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "=======================================================\n";
    return 0;
}


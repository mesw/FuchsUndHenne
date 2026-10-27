#ifndef FUCHS_UND_HENNE_BITBOARD_HPP
#define FUCHS_UND_HENNE_BITBOARD_HPP

#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <tuple>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace fuchs {

constexpr int NUM_NODES = 33;
constexpr int NUM_DIRECTIONS = 16;
constexpr int NUM_ACTIONS = NUM_NODES * NUM_DIRECTIONS; // 528
constexpr int GRID_SIZE = 7;

// 64-bit mask of all 33 valid board nodes: bits 0..32 set to 1
constexpr uint64_t ALL_BOARD_NODES_MASK = (1ULL << NUM_NODES) - 1ULL;

// The 9 Stall nodes: 0, 1, 2, 3, 4, 5, 8, 9, 10
constexpr uint64_t STALL_NODES_MASK = 
    (1ULL << 0) | (1ULL << 1) | (1ULL << 2) |
    (1ULL << 3) | (1ULL << 4) | (1ULL << 5) |
    (1ULL << 8) | (1ULL << 9) | (1ULL << 10);

// Initial piece bitboards
constexpr uint64_t START_FOXES_MASK = (1ULL << 8) | (1ULL << 10); // Nodes (2,2) and (2,4)

// 20 Hens: Nodes 13 through 32
constexpr uint64_t START_HENS_MASK = 
    (1ULL << 13) | (1ULL << 14) | (1ULL << 15) | (1ULL << 16) | (1ULL << 17) |
    (1ULL << 18) | (1ULL << 19) | (1ULL << 20) | (1ULL << 21) | (1ULL << 22) |
    (1ULL << 23) | (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27) |
    (1ULL << 28) | (1ULL << 29) | (1ULL << 30) | (1ULL << 31) | (1ULL << 32);

// Board node coordinates (row, col)
struct Coord {
    int8_t r;
    int8_t c;
};

constexpr std::array<Coord, NUM_NODES> NODE_COORDS = {{
    {0, 2}, {0, 3}, {0, 4},
    {1, 2}, {1, 3}, {1, 4},
    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6},
    {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6},
    {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
    {5, 2}, {5, 3}, {5, 4},
    {6, 2}, {6, 3}, {6, 4}
}};

// Precomputed lookup tables
inline const auto& get_step_targets() {
    static const auto targets = []() {
        std::array<std::array<int8_t, 8>, NUM_NODES> tbl{};
        for (int u = 0; u < NUM_NODES; ++u) {
            tbl[u].fill(-1);
        }

        const int dr[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        const int dc[8] = {0, 0, -1, 1, -1, 1, -1, 1};

        auto find_node = [](int r, int c) -> int {
            for (int i = 0; i < NUM_NODES; ++i) {
                if (NODE_COORDS[i].r == r && NODE_COORDS[i].c == c) return i;
            }
            return -1;
        };

        for (int u = 0; u < NUM_NODES; ++u) {
            int r = NODE_COORDS[u].r;
            int c = NODE_COORDS[u].c;
            for (int d = 0; d < 8; ++d) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                int target = find_node(nr, nc);
                if (target != -1) {
                    if (d < 4) {
                        tbl[u][d] = static_cast<int8_t>(target);
                    } else {
                        if (find_node(r, nc) != -1 && find_node(nr, c) != -1) {
                            tbl[u][d] = static_cast<int8_t>(target);
                        }
                    }
                }
            }
        }
        return tbl;
    }();
    return targets;
}

inline const auto& get_jump_tables() {
    static const auto tables = []() {
        std::array<std::array<int8_t, 8>, NUM_NODES> over{};
        std::array<std::array<int8_t, 8>, NUM_NODES> land{};
        const auto& step_targets = get_step_targets();

        for (int u = 0; u < NUM_NODES; ++u) {
            over[u].fill(-1);
            land[u].fill(-1);
            for (int d = 0; d < 8; ++d) {
                int8_t mid = step_targets[u][d];
                if (mid != -1) {
                    int8_t dst = step_targets[mid][d];
                    if (dst != -1) {
                        over[u][d] = mid;
                        land[u][d] = dst;
                    }
                }
            }
        }
        return std::make_pair(over, land);
    }();
    return tables;
}

inline int count_bits(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    int count = 0;
    while (x) { count += (x & 1ULL); x >>= 1ULL; }
    return count;
#endif
}

struct BitBoard {
    uint64_t foxes{START_FOXES_MASK};
    uint64_t hens{START_HENS_MASK};
    bool fox_turn{false}; // Hens move first
    int turn_count{0};
    int active_jumping_fox{-1};

    void reset() noexcept {
        foxes = START_FOXES_MASK;
        hens = START_HENS_MASK;
        fox_turn = false;
        turn_count = 0;
        active_jumping_fox = -1;
    }

    uint64_t occupied() const noexcept {
        return foxes | hens;
    }

    uint64_t empty() const noexcept {
        return (~(foxes | hens)) & ALL_BOARD_NODES_MASK;
    }

    int hen_count() const noexcept {
        return count_bits(hens);
    }

    int fox_count() const noexcept {
        return count_bits(foxes);
    }
};

inline constexpr int encode_action(int origin_node, int direction_id) noexcept {
    return origin_node * NUM_DIRECTIONS + direction_id;
}

inline constexpr std::pair<int, int> decode_action(int action) noexcept {
    return {action / NUM_DIRECTIONS, action % NUM_DIRECTIONS};
}

inline std::vector<int> get_legal_actions(const BitBoard& state) {
    std::vector<int> actions;
    actions.reserve(64);

    const auto& steps = get_step_targets();
    const auto& [jump_over, jump_land] = get_jump_tables();
    const uint64_t occ = state.occupied();

    if (!state.fox_turn) {
        constexpr int hen_dirs[5] = {0, 2, 3, 4, 5};
        uint64_t h_bits = state.hens;

        while (h_bits) {
            int u;
            #if defined(__GNUC__) || defined(__clang__)
            u = __builtin_ctzll(h_bits);
            #elif defined(_MSC_VER)
            unsigned long idx;
            _BitScanForward64(&idx, h_bits);
            u = static_cast<int>(idx);
            #else
            u = 0;
            while (!((h_bits >> u) & 1ULL)) ++u;
            #endif
            h_bits &= ~(1ULL << u);

            for (int d : hen_dirs) {
                int8_t dst = steps[u][d];
                if (dst != -1 && !((occ >> dst) & 1ULL)) {
                    actions.push_back(encode_action(u, d));
                }
            }
        }
    } else {
        std::vector<int> step_actions;
        std::vector<int> jump_actions;

        uint64_t f_bits = state.foxes;
        if (state.active_jumping_fox != -1) {
            f_bits = (1ULL << state.active_jumping_fox);
        }

        while (f_bits) {
            int u;
            #if defined(__GNUC__) || defined(__clang__)
            u = __builtin_ctzll(f_bits);
            #elif defined(_MSC_VER)
            unsigned long idx;
            _BitScanForward64(&idx, f_bits);
            u = static_cast<int>(idx);
            #else
            u = 0;
            while (!((f_bits >> u) & 1ULL)) ++u;
            #endif
            f_bits &= ~(1ULL << u);

            for (int d = 0; d < 8; ++d) {
                if (state.active_jumping_fox == -1) {
                    int8_t dst = steps[u][d];
                    if (dst != -1 && !((occ >> dst) & 1ULL)) {
                        step_actions.push_back(encode_action(u, d));
                    }
                }

                int8_t mid = jump_over[u][d];
                int8_t land = jump_land[u][d];
                if (mid != -1 && ((state.hens >> mid) & 1ULL) && !((occ >> land) & 1ULL)) {
                    jump_actions.push_back(encode_action(u, d + 8));
                }
            }
        }

        actions = std::move(jump_actions);
        if (state.active_jumping_fox == -1) {
            actions.insert(actions.end(), step_actions.begin(), step_actions.end());
        }
    }

    return actions;
}

inline bool apply_action(BitBoard& state, int action) {
    const auto [origin, dir] = decode_action(action);
    const auto& steps = get_step_targets();
    const auto& [jump_over, jump_land] = get_jump_tables();

    bool is_capture = false;

    if (!state.fox_turn) {
        if (dir >= 8) return false;
        int8_t dst = steps[origin][dir];
        if (dst == -1 || ((state.occupied() >> dst) & 1ULL)) return false;

        state.hens &= ~(1ULL << origin);
        state.hens |= (1ULL << dst);
        state.fox_turn = true;
        state.turn_count++;
        state.active_jumping_fox = -1;
    } else {
        if (dir < 8) {
            int8_t dst = steps[origin][dir];
            if (dst == -1 || ((state.occupied() >> dst) & 1ULL)) return false;

            state.foxes &= ~(1ULL << origin);
            state.foxes |= (1ULL << dst);
            state.fox_turn = false;
            state.turn_count++;
            state.active_jumping_fox = -1;
        } else {
            int step_d = dir - 8;
            int8_t mid = jump_over[origin][step_d];
            int8_t land = jump_land[origin][step_d];
            if (mid == -1 || land == -1) return false;

            state.foxes &= ~(1ULL << origin);
            state.foxes |= (1ULL << land);
            state.hens &= ~(1ULL << mid);
            is_capture = true;

            bool has_next_jump = false;
            const uint64_t occ = state.occupied();
            for (int d = 0; d < 8; ++d) {
                int8_t m = jump_over[land][d];
                int8_t l = jump_land[land][d];
                if (m != -1 && ((state.hens >> m) & 1ULL) && !((occ >> l) & 1ULL)) {
                    has_next_jump = true;
                    break;
                }
            }

            if (has_next_jump) {
                state.active_jumping_fox = land;
            } else {
                state.active_jumping_fox = -1;
                state.fox_turn = false;
                state.turn_count++;
            }
        }
    }
    return true;
}

inline std::tuple<bool, int, std::string> check_game_over(const BitBoard& state, int max_turns = 250) {
    if ((state.hens & STALL_NODES_MASK) == STALL_NODES_MASK) {
        return {true, 1, "stall_filled"};
    }

    BitBoard fox_test = state;
    fox_test.fox_turn = true;
    fox_test.active_jumping_fox = -1;
    if (get_legal_actions(fox_test).empty()) {
        return {true, 1, "foxes_trapped"};
    }

    BitBoard hen_test = state;
    hen_test.fox_turn = false;
    if (get_legal_actions(hen_test).empty()) {
        return {true, -1, "hens_trapped"};
    }

    if (state.hen_count() <= 8) {
        return {true, -1, "hens_depleted"};
    }

    if (max_turns > 0 && state.turn_count >= max_turns) {
        return {true, 0, "max_turns_reached"};
    }

    return {false, 0, ""};
}

} // namespace fuchs

#endif // FUCHS_UND_HENNE_BITBOARD_HPP

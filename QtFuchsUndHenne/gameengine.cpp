#include "gameengine.hpp"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QDebug>
#include <iostream>
#include <algorithm>

GameEngine::GameEngine(QObject *parent)
    : QObject(parent)
{
    m_board.reset();
    loadModels();

    connect(&m_aiTimer, &QTimer::timeout, this, &GameEngine::onAiTimerTimeout);
    updateAiTimer();
}

QString GameEngine::modeName() const {
    switch (m_mode) {
    case 0: return QStringLiteral("AI vs AI (Spectator)");
    case 1: return QStringLiteral("Human (Hens) vs AI (Fox)");
    case 2: return QStringLiteral("Human (Fox) vs AI (Hens)");
    case 3: return QStringLiteral("Human vs Human (Local)");
    default: return QStringLiteral("Unknown");
    }
}

void GameEngine::setMode(int mode) {
    if (m_mode == mode) return;
    m_mode = mode;
    m_selectedNode = -1;
    m_legalDestinations.clear();
    emit modeChanged();
    emit selectionChanged();
    updateAiTimer();
}

void GameEngine::nextMode() {
    setMode((m_mode + 1) % 4);
}

void GameEngine::setPaused(bool paused) {
    if (m_isPaused == paused) return;
    m_isPaused = paused;
    emit pausedChanged();
    updateAiTimer();
}

void GameEngine::togglePause() {
    setPaused(!m_isPaused);
}

void GameEngine::setSpeedFps(int fps) {
    fps = std::clamp(fps, 1, 30);
    if (m_speedFps == fps) return;
    m_speedFps = fps;
    emit speedChanged();
    updateAiTimer();
}

void GameEngine::setMaxTurns(int turns) {
    if (m_maxTurns == turns) return;
    m_maxTurns = turns;
    emit maxTurnsChanged();
}

QString GameEngine::winnerString() const {
    if (!m_gameOver) return QString();
    if (m_winner == 1) return QStringLiteral("HENS WIN!");
    if (m_winner == -1) return QStringLiteral("FOXES WIN!");
    return QStringLiteral("DRAW!");
}

bool GameEngine::isAiTurn() const {
    if (m_gameOver) return false;
    if (m_mode == 0) return true;
    if (m_mode == 1 && m_board.fox_turn) return true;
    if (m_mode == 2 && !m_board.fox_turn) return true;
    return false;
}

void GameEngine::updateAiTimer() {
    if (!m_isPaused && !m_gameOver && isAiTurn()) {
        int intervalMs = std::max(20, 1000 / m_speedFps);
        if (!m_aiTimer.isActive() || m_aiTimer.interval() != intervalMs) {
            m_aiTimer.start(intervalMs);
        }
    } else {
        m_aiTimer.stop();
    }
}

void GameEngine::onAiTimerTimeout() {
    if (!m_isPaused && !m_gameOver && isAiTurn()) {
        stepAi();
    }
}

void GameEngine::resetGame() {
    m_board.reset();
    m_selectedNode = -1;
    m_legalDestinations.clear();
    m_lastHenSrc = -1;
    m_lastHenDst = -1;
    m_consecutiveLeerzuege = 0;
    m_henBoardHistory.clear();
    m_leerzugWarning.clear();
    m_gameOver = false;
    m_winner = 0;
    m_terminationReason.clear();

    emit stateChanged();
    emit selectionChanged();
    emit gameOverChanged();
    emit leerzugChanged();
    updateAiTimer();
}

std::vector<int> GameEngine::filterHenLegalActions(const std::vector<int>& actions, const fuchs::BitBoard& board) const {
    if (board.fox_turn) return actions;

    const auto& steps = fuchs::get_step_targets();
    std::vector<int> filtered;

    for (int act : actions) {
        auto [orig, dir] = fuchs::decode_action(act);
        if (dir >= 8) continue; // Hens cannot jump
        int dest = steps[orig][dir];
        if (dest == -1) continue;

        // 1. Traditional 3-Leerzüge rule: "an der gleichen Position maximal 3 Leerzüge hin und her fahren"
        // After 3 consecutive back-and-forth moves between m_lastHenSrc and m_lastHenDst, reversing back is illegal!
        if (m_consecutiveLeerzuege >= 3 && orig == m_lastHenDst && dest == m_lastHenSrc) {
            continue; // Prohibited by 3-Leerzüge limit!
        }

        // 2. Multi-hen repetition protection: prevent 3rd occurrence of identical hen formation
        uint64_t next_hens = (board.hens & ~(1ULL << orig)) | (1ULL << dest);
        int repeats = 0;
        for (uint64_t h_state : m_henBoardHistory) {
            if (h_state == next_hens) repeats++;
        }
        if (repeats >= 2) {
            continue; // Prohibited: would create a 3rd identical hen formation!
        }

        filtered.push_back(act);
    }

    return filtered;
}

void GameEngine::recordHenMove(int src, int dst) {
    int r_src = fuchs::NODE_COORDS[src].r;
    int r_dst = fuchs::NODE_COORDS[dst].r;

    if (r_dst < r_src) {
        // Forward move: irreversible progress towards stall!
        m_consecutiveLeerzuege = 0;
        m_lastHenSrc = -1;
        m_lastHenDst = -1;
        m_henBoardHistory.clear();
        m_leerzugWarning.clear();
    } else {
        // Horizontal / sideways move (Leerzug)
        if (src == m_lastHenDst && dst == m_lastHenSrc) {
            m_consecutiveLeerzuege++;
        } else {
            m_consecutiveLeerzuege = 1;
        }
        m_lastHenSrc = src;
        m_lastHenDst = dst;

        if (m_consecutiveLeerzuege >= 3) {
            m_leerzugWarning = QStringLiteral("3 Leerzüge an gleicher Position erreicht! Vorwärtszug oder andere Henne erforderlich.");
        } else if (m_consecutiveLeerzuege == 2) {
            m_leerzugWarning = QStringLiteral("Leerzug 2/3 an gleicher Position");
        } else {
            m_leerzugWarning = QStringLiteral("Leerzug 1/3");
        }
    }
    m_henBoardHistory.push_back(m_board.hens);
    emit leerzugChanged();
}

QVariantList GameEngine::nodeStates() const {
    QVariantList list;
    list.reserve(fuchs::NUM_NODES);
    for (int i = 0; i < fuchs::NUM_NODES; ++i) {
        if ((m_board.foxes >> i) & 1ULL) {
            list.append(2); // Fox
        } else if ((m_board.hens >> i) & 1ULL) {
            list.append(1); // Hen
        } else {
            list.append(0); // Empty
        }
    }
    return list;
}

QVariantList GameEngine::legalDestinations() const {
    QVariantList list;
    for (const auto& [dest, act] : m_legalDestinations) {
        list.append(dest);
    }
    return list;
}

QVariantList GameEngine::getAllEdges() const {
    QVariantList edges;
    const auto& steps = fuchs::get_step_targets();
    for (int u = 0; u < fuchs::NUM_NODES; ++u) {
        int r1 = fuchs::NODE_COORDS[u].r;
        int c1 = fuchs::NODE_COORDS[u].c;
        for (int d = 0; d < 8; ++d) {
            int v = steps[u][d];
            if (v != -1 && v > u) {
                int r2 = fuchs::NODE_COORDS[v].r;
                int c2 = fuchs::NODE_COORDS[v].c;
                QVariantMap edge;
                edge[QStringLiteral("u")] = u;
                edge[QStringLiteral("v")] = v;
                edge[QStringLiteral("r1")] = r1;
                edge[QStringLiteral("c1")] = c1;
                edge[QStringLiteral("r2")] = r2;
                edge[QStringLiteral("c2")] = c2;
                edges.append(edge);
            }
        }
    }
    return edges;
}

QVariantList GameEngine::getNodeCoords() const {
    QVariantList nodes;
    for (int u = 0; u < fuchs::NUM_NODES; ++u) {
        QVariantMap node;
        node[QStringLiteral("index")] = u;
        node[QStringLiteral("r")] = fuchs::NODE_COORDS[u].r;
        node[QStringLiteral("c")] = fuchs::NODE_COORDS[u].c;
        node[QStringLiteral("isStall")] = bool((fuchs::STALL_NODES_MASK >> u) & 1ULL);
        nodes.append(node);
    }
    return nodes;
}

void GameEngine::updateSelectionForActiveJump() {
    if (m_board.active_jumping_fox != -1) {
        m_selectedNode = m_board.active_jumping_fox;
        m_legalDestinations.clear();
        auto legal_actions = fuchs::get_legal_actions(m_board);
        const auto& [jump_over, jump_land] = fuchs::get_jump_tables();
        for (int act : legal_actions) {
            auto [orig, dir] = fuchs::decode_action(act);
            if (orig == m_selectedNode && dir >= 8) {
                int step_dir = dir - 8;
                int dest = jump_land[orig][step_dir];
                if (dest != -1) {
                    m_legalDestinations[dest] = act;
                }
            }
        }
        emit selectionChanged();
    }
}

void GameEngine::nodeClicked(int nodeIndex) {
    if (m_gameOver) return;

    // Determine if human is allowed to move this turn
    bool humanTurn = false;
    if (m_mode == 3) {
        humanTurn = true; // Human vs Human
    } else if (m_mode == 1 && !m_board.fox_turn) {
        humanTurn = true; // Human as Hens
    } else if (m_mode == 2 && m_board.fox_turn) {
        humanTurn = true; // Human as Fox
    }

    if (!humanTurn) return;

    // Check if clicked node is a legal move destination
    if (m_selectedNode != -1 && m_legalDestinations.count(nodeIndex)) {
        int action = m_legalDestinations[nodeIndex];
        auto [orig, dir] = fuchs::decode_action(action);
        bool wasHen = !m_board.fox_turn;
        bool wasFoxCapture = m_board.fox_turn && (dir >= 8);

        fuchs::apply_action(m_board, action);

        if (wasHen) {
            recordHenMove(orig, nodeIndex);
        } else if (wasFoxCapture) {
            m_consecutiveLeerzuege = 0;
            m_lastHenSrc = -1;
            m_lastHenDst = -1;
            m_henBoardHistory.clear();
            m_leerzugWarning.clear();
            emit leerzugChanged();
        }

        m_selectedNode = -1;
        m_legalDestinations.clear();

        // Check terminal state
        auto [done, win, reason] = fuchs::check_game_over(m_board, m_maxTurns);
        if (done) {
            m_gameOver = true;
            m_winner = win;
            m_terminationReason = QString::fromStdString(reason);
            emit gameOverChanged();
        } else {
            // If fox has a chained multi-jump, auto-select it
            if (m_board.active_jumping_fox != -1) {
                updateSelectionForActiveJump();
            }
        }

        emit stateChanged();
        emit selectionChanged();
        updateAiTimer();
        return;
    }

    // If fox is mid-chain jump, cannot select another piece!
    if (m_board.active_jumping_fox != -1) {
        // Clicking active fox piece concludes the multi-jump chain voluntarily
        if (nodeIndex == m_board.active_jumping_fox) {
            m_board.active_jumping_fox = -1;
            m_board.fox_turn = false;
            m_board.turn_count++;
            m_selectedNode = -1;
            m_legalDestinations.clear();
            emit stateChanged();
            emit selectionChanged();
            updateAiTimer();
            return;
        }
        updateSelectionForActiveJump();
        return;
    }

    // Select piece belonging to active player
    bool isPieceOfCurrentPlayer = false;
    if (m_board.fox_turn && ((m_board.foxes >> nodeIndex) & 1ULL)) {
        isPieceOfCurrentPlayer = true;
    } else if (!m_board.fox_turn && ((m_board.hens >> nodeIndex) & 1ULL)) {
        isPieceOfCurrentPlayer = true;
    }

    if (isPieceOfCurrentPlayer) {
        m_selectedNode = nodeIndex;
        m_legalDestinations.clear();

        auto legal_actions = fuchs::get_legal_actions(m_board);
        if (!m_board.fox_turn) {
            legal_actions = filterHenLegalActions(legal_actions, m_board);
        }
        const auto& steps = fuchs::get_step_targets();
        const auto& [jump_over, jump_land] = fuchs::get_jump_tables();

        for (int act : legal_actions) {
            auto [orig, dir] = fuchs::decode_action(act);
            if (orig == nodeIndex) {
                int dest = -1;
                if (dir < 8) {
                    dest = steps[orig][dir];
                } else {
                    int step_dir = dir - 8;
                    dest = jump_land[orig][step_dir];
                }
                if (dest != -1) {
                    m_legalDestinations[dest] = act;
                }
            }
        }
        emit selectionChanged();
    } else {
        // Deselect
        m_selectedNode = -1;
        m_legalDestinations.clear();
        emit selectionChanged();
    }
}

void GameEngine::stepAi() {
    if (m_gameOver) return;

    auto raw_actions = fuchs::get_legal_actions(m_board);
    std::vector<int> legal_actions;
    if (!m_board.fox_turn) {
        legal_actions = filterHenLegalActions(raw_actions, m_board);
    } else {
        legal_actions = raw_actions;
    }

    if (legal_actions.empty()) {
        auto [done, win, reason] = fuchs::check_game_over(m_board, m_maxTurns);
        m_gameOver = true;
        m_winner = win != 0 ? win : (m_board.fox_turn ? 1 : -1);
        m_terminationReason = QString::fromStdString(reason.empty() ? (m_board.fox_turn ? "foxes_trapped" : "hens_trapped") : reason);
        emit gameOverChanged();
        emit stateChanged();
        updateAiTimer();
        return;
    }

    fuchs::ObsTensor obs = fuchs::make_observation_tensor(m_board);
    int chosen_action = -1;
    if (m_board.fox_turn) {
        chosen_action = fuchs::predict_best_action(m_foxWeights, obs, legal_actions);
    } else {
        // Tactical Guard for Hens: Filter out blunder moves that leave immediate jumps for the Fox
        std::vector<int> safe_hen_actions;
        for (int act : legal_actions) {
            fuchs::BitBoard next_b = m_board;
            fuchs::apply_action(next_b, act);
            auto fox_moves = fuchs::get_legal_actions(next_b);
            bool gives_immediate_jump = false;
            for (int fa : fox_moves) {
                if (fuchs::decode_action(fa).second >= 8) {
                    gives_immediate_jump = true;
                    break;
                }
            }
            if (!gives_immediate_jump) {
                safe_hen_actions.push_back(act);
            }
        }
        // If safe actions exist among non-repeating moves, use them;
        // if ALL non-repeating moves are under threat, Hen AI is FORCED to advance instead of repeating!
        const std::vector<int>& hen_actions = safe_hen_actions.empty() ? legal_actions : safe_hen_actions;
        chosen_action = fuchs::predict_best_action(m_henWeights, obs, hen_actions);
    }

    auto [orig, dir] = fuchs::decode_action(chosen_action);
    bool wasHen = !m_board.fox_turn;
    bool wasFoxCapture = m_board.fox_turn && (dir >= 8);
    int dest_node = -1;
    if (dir < 8) {
        dest_node = fuchs::get_step_targets()[orig][dir];
    } else {
        dest_node = fuchs::get_jump_tables().second[orig][dir - 8];
    }

    fuchs::apply_action(m_board, chosen_action);

    if (wasHen && dest_node != -1) {
        recordHenMove(orig, dest_node);
    } else if (wasFoxCapture) {
        m_consecutiveLeerzuege = 0;
        m_lastHenSrc = -1;
        m_lastHenDst = -1;
        m_henBoardHistory.clear();
        m_leerzugWarning.clear();
        emit leerzugChanged();
    }

    // Check terminal state
    auto [done, win, reason] = fuchs::check_game_over(m_board, m_maxTurns);
    if (done) {
        m_gameOver = true;
        m_winner = win;
        m_terminationReason = QString::fromStdString(reason);
        emit gameOverChanged();
    }

    emit stateChanged();
    updateAiTimer();
}

void GameEngine::stepOnce() {
    if (!m_gameOver && isAiTurn()) {
        stepAi();
    }
}

static QString findLatestModelFile(const QStringList& candidates) {
    QString bestPath;
    QDateTime bestTime;

    for (const QString& path : candidates) {
        QFileInfo fi(path);
        if (fi.exists() && fi.isFile() && fi.size() > 0) {
            QDateTime modTime = fi.lastModified();
            QString resolved = fi.canonicalFilePath();
            if (resolved.isEmpty()) {
                resolved = fi.absoluteFilePath();
            }
            if (bestPath.isEmpty() || modTime > bestTime) {
                bestPath = resolved;
                bestTime = modTime;
            }
        }
    }
    return bestPath;
}

void GameEngine::loadModels() {
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList foxCandidates = {
        QStringLiteral("/fox_actor.bin"),
        QStringLiteral("fox_actor.bin"),
        appDir + QStringLiteral("/fox_actor.bin"),
        appDir + QStringLiteral("/../fox_actor.bin"),
        appDir + QStringLiteral("/../../fox_actor.bin"),
        appDir + QStringLiteral("/../../../fox_actor.bin"),
        appDir + QStringLiteral("/../../../exported_models/fox_actor.bin"),
        appDir + QStringLiteral("/../../exported_models/fox_actor.bin"),
        appDir + QStringLiteral("/../exported_models/fox_actor.bin"),
        QStringLiteral("../fox_actor.bin"),
        QStringLiteral("../exported_models/fox_actor.bin"),
        QStringLiteral("../../exported_models/fox_actor.bin"),
        QStringLiteral("exported_models/fox_actor.bin")
    };

    QString latestFox = findLatestModelFile(foxCandidates);
    if (!latestFox.isEmpty() && m_foxWeights.load_from_binary(latestFox.toStdString())) {
        m_foxModelName = QFileInfo(latestFox).fileName();
        qDebug() << "[*] Loaded Fox model from latest file:" << latestFox << "(modified:" << QFileInfo(latestFox).lastModified().toString() << ")";
    } else {
        m_foxModelName = QStringLiteral("Random / Fallback");
    }

    QStringList henCandidates = {
        QStringLiteral("/hens_actor.bin"),
        QStringLiteral("hens_actor.bin"),
        appDir + QStringLiteral("/hens_actor.bin"),
        appDir + QStringLiteral("/../hens_actor.bin"),
        appDir + QStringLiteral("/../../hens_actor.bin"),
        appDir + QStringLiteral("/../../../hens_actor.bin"),
        appDir + QStringLiteral("/../../../exported_models/hens_actor.bin"),
        appDir + QStringLiteral("/../../exported_models/hens_actor.bin"),
        appDir + QStringLiteral("/../exported_models/hens_actor.bin"),
        QStringLiteral("../hens_actor.bin"),
        QStringLiteral("../exported_models/hens_actor.bin"),
        QStringLiteral("../../exported_models/hens_actor.bin"),
        QStringLiteral("exported_models/hens_actor.bin")
    };

    QString latestHen = findLatestModelFile(henCandidates);
    if (!latestHen.isEmpty() && m_henWeights.load_from_binary(latestHen.toStdString())) {
        m_henModelName = QFileInfo(latestHen).fileName();
        qDebug() << "[*] Loaded Hen model from latest file:" << latestHen << "(modified:" << QFileInfo(latestHen).lastModified().toString() << ")";
    } else {
        m_henModelName = QStringLiteral("Random / Fallback");
    }

    emit modelsLoaded();
}

void GameEngine::reloadModels() {
    loadModels();
}

#ifndef GAMEENGINE_HPP
#define GAMEENGINE_HPP

#include <QObject>
#include <QTimer>
#include <QVariantList>
#include <QVariantMap>
#include <QString>
#include <QtQml/qqmlregistration.h>
#include <map>
#include <vector>
#include <tuple>
#include "bitboard.hpp"
#include "inference.hpp"

class GameEngine : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(int mode READ mode WRITE setMode NOTIFY modeChanged)
    Q_PROPERTY(QString modeName READ modeName NOTIFY modeChanged)
    Q_PROPERTY(bool isFoxTurn READ isFoxTurn NOTIFY stateChanged)
    Q_PROPERTY(int turnCount READ turnCount NOTIFY stateChanged)
    Q_PROPERTY(int henCount READ henCount NOTIFY stateChanged)
    Q_PROPERTY(int foxCount READ foxCount NOTIFY stateChanged)
    Q_PROPERTY(bool gameOver READ gameOver NOTIFY gameOverChanged)
    Q_PROPERTY(int winner READ winner NOTIFY gameOverChanged)
    Q_PROPERTY(QString winnerString READ winnerString NOTIFY gameOverChanged)
    Q_PROPERTY(QString terminationReason READ terminationReason NOTIFY gameOverChanged)
    Q_PROPERTY(bool isPaused READ isPaused WRITE setPaused NOTIFY pausedChanged)
    Q_PROPERTY(int speedFps READ speedFps WRITE setSpeedFps NOTIFY speedChanged)
    Q_PROPERTY(int selectedNode READ selectedNode NOTIFY selectionChanged)
    Q_PROPERTY(QVariantList nodeStates READ nodeStates NOTIFY stateChanged)
    Q_PROPERTY(QVariantList legalDestinations READ legalDestinations NOTIFY selectionChanged)
    Q_PROPERTY(QString foxModelName READ foxModelName NOTIFY modelsLoaded)
    Q_PROPERTY(QString henModelName READ henModelName NOTIFY modelsLoaded)
    Q_PROPERTY(int maxTurns READ maxTurns WRITE setMaxTurns NOTIFY maxTurnsChanged)
    Q_PROPERTY(bool isAiTurn READ isAiTurn NOTIFY stateChanged)
    Q_PROPERTY(int activeJumpingFox READ activeJumpingFox NOTIFY stateChanged)
    Q_PROPERTY(int consecutiveLeerzuege READ consecutiveLeerzuege NOTIFY leerzugChanged)
    Q_PROPERTY(QString leerzugWarning READ leerzugWarning NOTIFY leerzugChanged)
    Q_PROPERTY(bool stochastic READ isStochastic WRITE setStochastic NOTIFY stochasticChanged)
    Q_PROPERTY(QString styleName READ styleName NOTIFY stochasticChanged)

public:
    explicit GameEngine(QObject *parent = nullptr);
    ~GameEngine() override = default;

    int mode() const { return m_mode; }
    void setMode(int mode);
    QString modeName() const;

    bool isFoxTurn() const { return m_board.fox_turn; }
    int turnCount() const { return m_board.turn_count; }
    int henCount() const { return m_board.hen_count(); }
    int foxCount() const { return m_board.fox_count(); }

    bool gameOver() const { return m_gameOver; }
    int winner() const { return m_winner; }
    QString winnerString() const;
    QString terminationReason() const { return m_terminationReason; }

    bool isPaused() const { return m_isPaused; }
    void setPaused(bool paused);

    int speedFps() const { return m_speedFps; }
    void setSpeedFps(int fps);

    int selectedNode() const { return m_selectedNode; }
    QVariantList nodeStates() const;
    QVariantList legalDestinations() const;

    QString foxModelName() const { return m_foxModelName; }
    QString henModelName() const { return m_henModelName; }

    int maxTurns() const { return m_maxTurns; }
    void setMaxTurns(int turns);

    bool isAiTurn() const;
    int activeJumpingFox() const { return m_board.active_jumping_fox; }
    int consecutiveLeerzuege() const { return m_consecutiveLeerzuege; }
    QString leerzugWarning() const { return m_leerzugWarning; }

    bool isStochastic() const { return m_stochastic; }
    void setStochastic(bool stochastic);
    QString styleName() const;
    Q_INVOKABLE void toggleStochastic();

    // Board Geometry helpers for QML
    Q_INVOKABLE QVariantList getAllEdges() const;
    Q_INVOKABLE QVariantList getNodeCoords() const;

    // Actions & Controls
    Q_INVOKABLE void resetGame();
    Q_INVOKABLE void togglePause();
    Q_INVOKABLE void nextMode();
    Q_INVOKABLE void nodeClicked(int nodeIndex);
    Q_INVOKABLE void stepAi();
    Q_INVOKABLE void stepOnce();
    Q_INVOKABLE void reloadModels();

signals:
    void stateChanged();
    void gameOverChanged();
    void modeChanged();
    void pausedChanged();
    void speedChanged();
    void selectionChanged();
    void modelsLoaded();
    void maxTurnsChanged();
    void leerzugChanged();
    void stochasticChanged();

private slots:
    void onAiTimerTimeout();

private:
    void loadModels();
    void updateAiTimer();
    void updateSelectionForActiveJump();
    std::vector<int> filterHenLegalActions(const std::vector<int>& actions, const fuchs::BitBoard& board) const;
    void recordHenMove(int src, int dst);
    bool wouldCauseRepetition(int act) const;
    static bool isHenAdvancement(int src, int dst);

    fuchs::BitBoard m_board;
    fuchs::ActorWeights m_foxWeights;
    fuchs::ActorWeights m_henWeights;

    int m_mode{0}; // 0: AI vs AI, 1: Human (Hens) vs AI, 2: Human (Fox) vs AI, 3: Human vs Human
    bool m_stochastic{true}; // true: Softmax temperature sampling (varied games), false: pure argmax
    bool m_isPaused{false};
    int m_speedFps{3}; // Default 3 moves / second
    int m_maxTurns{300};

    int m_selectedNode{-1};
    std::map<int, int> m_legalDestinations; // destination_node -> action_id

    // 3-Leerzüge rule tracking
    int m_trackMin{-1};
    int m_trackMax{-1};
    int m_leerzugHalfMoves{0};
    int m_consecutiveLeerzuege{0}; // 0 to 3 consecutive moves on current track
    QString m_leerzugWarning;
    std::vector<std::tuple<uint64_t, uint64_t, bool>> m_boardHistory; // (foxes, hens, fox_turn) for anti-repetition tracking

    bool m_gameOver{false};
    int m_winner{0}; // 1 = Hens, -1 = Fox, 0 = Draw
    QString m_terminationReason;

    QString m_foxModelName{"None"};
    QString m_henModelName{"None"};

    QTimer m_aiTimer;
};

#endif // GAMEENGINE_HPP

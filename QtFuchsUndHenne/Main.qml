import QtQuick
import QtQuick.Layouts
import QtFuchsUndHenne 1.0

Window {
    id: root
    width: 980
    height: 720
    minimumWidth: 320
    minimumHeight: 320
    visible: true
    title: qsTr("Fuchs und Henne - C++ Deep RL Engine & Spectator")
    color: "#1a1d24"

    // Responsive Mobile & Desktop Layout Properties
    readonly property bool isVertical: (root.width < 720) || (root.height > root.width * 1.05)
    readonly property bool isNarrow: root.width <= 380
    readonly property real layoutMargin: isVertical ? (isNarrow ? 8 : 12) : 16
    readonly property real layoutSpacing: isVertical ? (isNarrow ? 8 : 12) : 16
    readonly property real cardPadding: isNarrow ? 8 : 12

    // Instantiate C++ GameEngine
    GameEngine {
        id: game
    }

    // Color Palette
    readonly property color cBg: "#1a1d24"
    readonly property color cBoardBg: "#232731"
    readonly property color cPanelBg: "#20242e"
    readonly property color cCardBg: "#272c38"
    readonly property color cLine: "#464e60"
    readonly property color cStallBg: "#2d3444"
    readonly property color cStallBorder: "#648cd2"
    readonly property color cNodeEmpty: "#373e4e"
    readonly property color cNodeBorder: "#5a6478"
    readonly property color cFox: "#eb4b4b"
    readonly property color cFoxRing: "#ff7878"
    readonly property color cHen: "#f5be41"
    readonly property color cHenRing: "#ffdc78"
    readonly property color cSelect: "#3cdc82"
    readonly property color cTextMain: "#f0f3f8"
    readonly property color cTextMuted: "#919baf"
    readonly property color cAccent: "#50a0ff"

    // Keyboard Hotkeys
    Item {
        focus: true
        Keys.onSpacePressed: game.togglePause()
        Keys.onEscapePressed: game.togglePause()
        Keys.onBackPressed: game.resetGame()
        Keys.onUpPressed: game.speedFps = Math.min(20, game.speedFps + 1)
        Keys.onDownPressed: game.speedFps = Math.max(1, game.speedFps - 1)
        Keys.onPressed: (event) => {
            if (event.key === Qt.Key_R) {
                game.resetGame()
            } else if (event.key === Qt.Key_M) {
                game.nextMode()
            } else if (event.key === Qt.Key_L) {
                game.reloadModels()
            } else if (event.key === Qt.Key_S) {
                game.stepOnce()
            } else if (event.key === Qt.Key_T) {
                game.toggleStochastic()
            }
        }
        Component.onCompleted: forceActiveFocus()
    }

    // Main Container: Flickable for smooth mobile scrolling when needed, locked on desktop
    Flickable {
        id: mainFlickable
        anchors.fill: parent
        contentWidth: width
        contentHeight: root.isVertical ? (mainLayout.implicitHeight + root.layoutMargin * 2) : height
        interactive: root.isVertical && (contentHeight > height)
        boundsBehavior: Flickable.StopAtBounds
        clip: true

        WheelHandler {
            target: mainFlickable
            acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
            onWheel: (event) => {
                if (mainFlickable.interactive) {
                    mainFlickable.contentY = Math.max(
                        0,
                        Math.min(mainFlickable.contentHeight - mainFlickable.height,
                                 mainFlickable.contentY - event.angleDelta.y)
                    )
                }
            }
        }

        // Responsive Grid: 1 column in vertical mobile mode, 2 columns in horizontal desktop mode
        GridLayout {
            id: mainLayout
            x: root.layoutMargin
            y: root.layoutMargin
            width: mainFlickable.width - root.layoutMargin * 2
            height: root.isVertical ? undefined : (mainFlickable.height - root.layoutMargin * 2)

            columns: root.isVertical ? 1 : 2
            rows: root.isVertical ? 2 : 1
            rowSpacing: root.layoutSpacing
            columnSpacing: root.layoutSpacing

            // -------------------------------------------------------------
            // Board Container (Top in Vertical, Left in Horizontal)
            // -------------------------------------------------------------
            Rectangle {
                id: boardContainer
                Layout.fillWidth: true
                Layout.fillHeight: !root.isVertical
                Layout.preferredHeight: root.isVertical ? (boardArea.boardSize + (root.isNarrow ? 20 : 32)) : -1
                color: cBoardBg
                radius: root.isNarrow ? 10 : 16
                border.color: "#2f3543"
                border.width: 1
                clip: true

                // Stall Label Badge
                Rectangle {
                    id: stallBadge
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.margins: root.isNarrow ? 8 : 12
                    height: root.isNarrow ? 20 : 26
                    width: stallLabel.implicitWidth + (root.isNarrow ? 10 : 16)
                    color: "#283142"
                    radius: 6
                    border.color: cStallBorder
                    border.width: 1
                    z: 10

                    Text {
                        id: stallLabel
                        anchors.centerIn: parent
                        text: root.isNarrow ? qsTr("STALL") : qsTr("STALL (9 NODES)")
                        color: "#a4c2f4"
                        font.pixelSize: root.isNarrow ? 10 : 11
                        font.bold: true
                    }
                }

                // Board View
                Item {
                    id: boardArea
                    readonly property real availW: Math.max(180, boardContainer.width - (root.isNarrow ? 16 : 32))
                    readonly property real availH: root.isVertical
                        ? Math.min(Math.max(220, root.height * 0.46), 480)
                        : Math.max(180, boardContainer.height - (root.isNarrow ? 16 : 32))
                    readonly property real boardSize: Math.max(200, Math.min(availW, availH, 620))
                    readonly property real cellSize: boardSize / 7.0
                    width: boardSize
                    height: boardSize
                    anchors.centerIn: parent

                    // Canvas for Grid Lattice and Stall highlight
                    Canvas {
                        id: gridCanvas
                        anchors.fill: parent

                        property var edges: game.getAllEdges()
                        property var nodes: game.getNodeCoords()
                        property real cs: boardArea.cellSize

                    onCsChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onHeightChanged: requestPaint()
                    Component.onCompleted: requestPaint()

                    onPaint: {
                        var ctx = getContext("2d");
                        if (!ctx) return;
                        ctx.clearRect(0, 0, width, height);

                        // 1. Draw Stall Area (Rows 0, 1, 2 and Cols 2, 3, 4)
                        var stallX = 2 * cs;
                        var stallY = 0 * cs;
                        var stallW = 3 * cs;
                        var stallH = 3 * cs;

                        ctx.fillStyle = cStallBg;
                        ctx.fillRect(stallX, stallY, stallW, stallH);

                        ctx.strokeStyle = cStallBorder;
                        ctx.lineWidth = 2;
                        ctx.strokeRect(stallX, stallY, stallW, stallH);

                        // 2. Draw Connection Lines (Orthogonal & Diagonal lattice)
                        ctx.strokeStyle = cLine;
                        ctx.lineWidth = 2.5;
                        ctx.lineCap = "round";

                        for (var i = 0; i < edges.length; ++i) {
                            var e = edges[i];
                            var x1 = e.c1 * cs + cs / 2.0;
                            var y1 = e.r1 * cs + cs / 2.0;
                            var x2 = e.c2 * cs + cs / 2.0;
                            var y2 = e.r2 * cs + cs / 2.0;

                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();
                        }

                        // 3. Draw Empty Node Bases
                        for (var j = 0; j < nodes.length; ++j) {
                            var n = nodes[j];
                            var nx = n.c * cs + cs / 2.0;
                            var ny = n.r * cs + cs / 2.0;

                            ctx.beginPath();
                            ctx.arc(nx, ny, 7, 0, 2 * Math.PI);
                            ctx.fillStyle = cNodeEmpty;
                            ctx.fill();
                            ctx.strokeStyle = cNodeBorder;
                            ctx.lineWidth = 1.5;
                            ctx.stroke();
                        }
                    }
                }

                // Interactive Piece and Destination Layer
                Repeater {
                    model: game.getNodeCoords()

                    Item {
                        id: nodeItem
                        required property var modelData
                        required property int index

                        readonly property real cs: boardArea.cellSize
                        readonly property real centerX: modelData.c * cs + cs / 2.0
                        readonly property real centerY: modelData.r * cs + cs / 2.0

                        x: centerX - width / 2.0
                        y: centerY - height / 2.0
                        width: cs
                        height: cs

                        readonly property int pieceState: game.nodeStates[index] || 0 // 0: Empty, 1: Hen, 2: Fox
                        readonly property bool isSelected: game.selectedNode === index
                        readonly property bool isLegalDest: game.legalDestinations.indexOf(index) !== -1

                        // Destination Indicator Ring (Pulsing green ring for valid moves)
                        Rectangle {
                            anchors.centerIn: parent
                            width: cs * 0.55
                            height: cs * 0.55
                            radius: width / 2.0
                            color: "transparent"
                            border.color: cSelect
                            border.width: 3
                            visible: nodeItem.isLegalDest

                            Rectangle {
                                anchors.centerIn: parent
                                width: parent.width * 0.35
                                height: parent.height * 0.35
                                radius: width / 2.0
                                color: cSelect
                            }

                            SequentialAnimation on scale {
                                running: nodeItem.isLegalDest
                                loops: Animation.Infinite
                                NumberAnimation { from: 0.9; to: 1.15; duration: 600; easing.type: Easing.InOutQuad }
                                NumberAnimation { from: 1.15; to: 0.9; duration: 600; easing.type: Easing.InOutQuad }
                            }
                        }

                        // Selection Halo
                        Rectangle {
                            anchors.centerIn: parent
                            width: cs * 0.68
                            height: cs * 0.68
                            radius: width / 2.0
                            color: "transparent"
                            border.color: cSelect
                            border.width: 3
                            visible: nodeItem.isSelected && game.activeJumpingFox !== index

                            SequentialAnimation on scale {
                                running: nodeItem.isSelected && game.activeJumpingFox !== index
                                loops: Animation.Infinite
                                NumberAnimation { from: 1.0; to: 1.12; duration: 500; easing.type: Easing.InOutQuad }
                                NumberAnimation { from: 1.12; to: 1.0; duration: 500; easing.type: Easing.InOutQuad }
                            }
                        }

                        // Multi-Jump Golden Halo
                        Rectangle {
                            anchors.centerIn: parent
                            width: cs * 0.74
                            height: cs * 0.74
                            radius: width / 2.0
                            color: "transparent"
                            border.color: "#ffe146"
                            border.width: 3.5
                            visible: game.activeJumpingFox === index

                            SequentialAnimation on scale {
                                running: game.activeJumpingFox === index
                                loops: Animation.Infinite
                                NumberAnimation { from: 1.0; to: 1.18; duration: 400; easing.type: Easing.InOutQuad }
                                NumberAnimation { from: 1.18; to: 1.0; duration: 400; easing.type: Easing.InOutQuad }
                            }
                        }

                        // Hen Piece (Gold)
                        Rectangle {
                            id: henPiece
                            anchors.centerIn: parent
                            width: cs * 0.52
                            height: cs * 0.52
                            radius: width / 2.0
                            color: cHen
                            border.color: cHenRing
                            border.width: 2.5
                            visible: nodeItem.pieceState === 1
                            scale: nodeMouse.containsMouse ? 1.1 : 1.0

                            Behavior on scale {
                                NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
                            }

                            // Subtle inner shine
                            Rectangle {
                                anchors.top: parent.top
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.topMargin: 3
                                width: parent.width * 0.45
                                height: parent.height * 0.25
                                radius: width / 2.0
                                color: "#ffffff"
                                opacity: 0.35
                            }
                        }

                        // Fox Piece (Crimson)
                        Rectangle {
                            id: foxPiece
                            anchors.centerIn: parent
                            width: cs * 0.60
                            height: cs * 0.60
                            radius: width / 2.0
                            color: cFox
                            border.color: cFoxRing
                            border.width: 3
                            visible: nodeItem.pieceState === 2
                            scale: nodeMouse.containsMouse ? 1.1 : 1.0

                            Behavior on scale {
                                NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
                            }

                            // Inner shine
                            Rectangle {
                                anchors.top: parent.top
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.topMargin: 4
                                width: parent.width * 0.5
                                height: parent.height * 0.25
                                radius: width / 2.0
                                color: "#ffffff"
                                opacity: 0.35
                            }
                        }

                        // Mouse Click Handler
                        MouseArea {
                            id: nodeMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: (nodeItem.pieceState > 0 || nodeItem.isLegalDest) ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: {
                                game.nodeClicked(nodeItem.index)
                            }
                        }
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // RIGHT / BOTTOM: Sidebar Information & Controls Panel
        // -------------------------------------------------------------
        Rectangle {
            id: sidebar
            Layout.fillWidth: root.isVertical
            Layout.preferredWidth: root.isVertical ? -1 : (root.width > 1100 ? 380 : 340)
            Layout.minimumWidth: root.isVertical ? -1 : 300
            Layout.maximumWidth: root.isVertical ? -1 : 420
            Layout.fillHeight: !root.isVertical
            Layout.preferredHeight: root.isVertical ? (sidebarContent.implicitHeight + (root.cardPadding * 2)) : -1
            color: cPanelBg
            radius: root.isNarrow ? 10 : 16
            border.color: "#2f3543"
            border.width: 1

            ColumnLayout {
                id: sidebarContent
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: root.isVertical ? undefined : parent.bottom
                anchors.margins: root.cardPadding
                spacing: root.isVertical ? (root.isNarrow ? 6 : 8) : 10

                // Header Title
                ColumnLayout {
                    spacing: 1
                    Text {
                        text: qsTr("Fuchs und Henne")
                        color: cTextMain
                        font.pixelSize: root.isNarrow ? 18 : 22
                        font.bold: true
                    }
                    Text {
                        text: qsTr("C++ Bitboard & Deep RL Spectator")
                        color: cTextMuted
                        font.pixelSize: root.isNarrow ? 11 : 12
                    }
                }

                // 1. Game Mode Selector Card
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: root.isNarrow ? 92 : 102
                    color: cCardBg
                    radius: 10
                    border.color: "#343b4b"
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: root.cardPadding
                        spacing: root.isNarrow ? 3 : 5

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("GAME MODE")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 10 : 11
                                font.bold: true
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                width: switchText.implicitWidth + (root.isNarrow ? 8 : 12)
                                height: root.isNarrow ? 20 : 22
                                radius: 4
                                color: modeMouse.containsMouse ? "#3b4457" : "#313848"
                                border.color: "#465268"
                                border.width: 1

                                Text {
                                    id: switchText
                                    anchors.centerIn: parent
                                    text: root.isVertical ? qsTr("Switch") : qsTr("Switch (M)")
                                    color: cAccent
                                    font.pixelSize: root.isNarrow ? 10 : 11
                                    font.bold: true
                                }
                                MouseArea {
                                    id: modeMouse
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.nextMode()
                                }
                            }
                        }

                        Text {
                            text: game.modeName
                            color: cAccent
                            font.pixelSize: root.isNarrow ? 13 : 14
                            font.bold: true
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                        // AI Play Style Row (Varied / Greedy)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 4

                            Text {
                                text: qsTr("AI Style:")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 10 : 11
                            }

                            Item { Layout.fillWidth: true }

                            Rectangle {
                                width: styleText.implicitWidth + (root.isNarrow ? 10 : 14)
                                height: root.isNarrow ? 20 : 22
                                radius: 4
                                color: game.stochastic ? "#1c3829" : (styleMouse.containsMouse ? "#3b4457" : "#313848")
                                border.color: game.stochastic ? cSelect : "#465268"
                                border.width: 1

                                Text {
                                    id: styleText
                                    anchors.centerIn: parent
                                    text: game.stochastic
                                        ? (root.isVertical ? qsTr("Varied") : qsTr("Varied (T)"))
                                        : (root.isVertical ? qsTr("Greedy") : qsTr("Greedy (T)"))
                                    color: game.stochastic ? cSelect : cTextMain
                                    font.pixelSize: root.isNarrow ? 10 : 11
                                    font.bold: true
                                }
                                MouseArea {
                                    id: styleMouse
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.toggleStochastic()
                                }
                            }
                        }
                    }
                }

                // 2. Status & Stats Card
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: game.consecutiveLeerzuege > 0 ? (root.isNarrow ? 146 : 168) : (root.isNarrow ? 116 : 132)
                    color: cCardBg
                    radius: 10
                    border.color: "#343b4b"
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: root.cardPadding
                        spacing: root.isNarrow ? 6 : 8

                        // Active Turn Banner
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Rectangle {
                                width: root.isNarrow ? 12 : 14
                                height: width
                                radius: width / 2
                                color: game.isFoxTurn ? cFox : cHen
                            }

                            Text {
                                text: game.isFoxTurn ? qsTr("ACTIVE: FOXES") : qsTr("ACTIVE: HENS")
                                color: game.isFoxTurn ? cFox : cHen
                                font.pixelSize: root.isNarrow ? 12 : 14
                                font.bold: true
                            }

                            Item { Layout.fillWidth: true }

                            // Multi-jump badge
                            Rectangle {
                                visible: game.activeJumpingFox !== -1
                                width: multiJumpText.implicitWidth + 8
                                height: root.isNarrow ? 18 : 20
                                radius: 4
                                color: "#541e1e"
                                border.color: cFox
                                border.width: 1

                                Text {
                                    id: multiJumpText
                                    anchors.centerIn: parent
                                    text: qsTr("MULTI-JUMP!")
                                    color: cFoxRing
                                    font.pixelSize: root.isNarrow ? 9 : 10
                                    font.bold: true
                                }
                            }
                        }

                        // Turn & Hens Stats
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("Turn Count:")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 12 : 13
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: game.turnCount + (game.maxTurns > 0 ? " / " + game.maxTurns : "")
                                color: cTextMain
                                font.pixelSize: root.isNarrow ? 12 : 13
                                font.bold: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("Hens Remaining:")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 12 : 13
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: game.henCount + " / 20"
                                color: game.henCount <= 8 ? cFox : cHen
                                font.pixelSize: root.isNarrow ? 12 : 13
                                font.bold: true
                            }
                        }

                        // 3-Leerzüge Rule Status Indicator
                        Rectangle {
                            visible: game.consecutiveLeerzuege > 0
                            Layout.fillWidth: true
                            Layout.preferredHeight: root.isNarrow ? 22 : 24
                            radius: 6
                            color: game.consecutiveLeerzuege >= 3 ? "#4d1c1c" : "#362e1c"
                            border.color: game.consecutiveLeerzuege >= 3 ? cFox : cHen
                            border.width: 1

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: root.isNarrow ? 6 : 8
                                anchors.rightMargin: root.isNarrow ? 6 : 8
                                spacing: 4

                                Text {
                                    text: game.consecutiveLeerzuege >= 3 ? "⚠ Leerzug: 3/3" : ("Leerzug: " + game.consecutiveLeerzuege + "/3")
                                    color: game.consecutiveLeerzuege >= 3 ? cFoxRing : cHenRing
                                    font.pixelSize: root.isNarrow ? 10 : 11
                                    font.bold: true
                                }

                                Item { Layout.fillWidth: true }

                                Text {
                                    text: game.consecutiveLeerzuege >= 3 ? qsTr("Anderer Zug Pflicht!") : qsTr("hin & her")
                                    color: game.consecutiveLeerzuege >= 3 ? "#ffaaaa" : cTextMuted
                                    font.pixelSize: root.isNarrow ? 9 : 10
                                    font.bold: game.consecutiveLeerzuege >= 3
                                }
                            }
                        }

                        // Hen Survival Progress Bar
                        Rectangle {
                            Layout.fillWidth: true
                            height: root.isNarrow ? 5 : 6
                            radius: height / 2
                            color: "#1d212b"

                            Rectangle {
                                width: parent.width * Math.min(1.0, game.henCount / 20.0)
                                height: parent.height
                                radius: parent.radius
                                color: game.henCount <= 8 ? cFox : cHen
                            }
                        }
                    }
                }

                // 3. AI Neural Networks Card
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: root.isNarrow ? 76 : 88
                    color: cCardBg
                    radius: 10
                    border.color: "#343b4b"
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: root.cardPadding
                        spacing: root.isNarrow ? 4 : 6

                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("AI MODELS")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 10 : 11
                                font.bold: true
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: root.isVertical ? qsTr("Reload") : qsTr("Reload (L)")
                                color: cAccent
                                font.pixelSize: root.isNarrow ? 10 : 11
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.reloadModels()
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Rectangle { width: 8; height: 8; radius: 4; color: cFox }
                            Text { text: qsTr("Fox Actor:"); color: cTextMuted; font.pixelSize: root.isNarrow ? 11 : 12 }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: game.foxModelName
                                color: cFox
                                font.pixelSize: root.isNarrow ? 11 : 12
                                font.bold: true
                                elide: Text.ElideMiddle
                                Layout.maximumWidth: parent.width * 0.55
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Rectangle { width: 8; height: 8; radius: 4; color: cHen }
                            Text { text: qsTr("Hens Actor:"); color: cTextMuted; font.pixelSize: root.isNarrow ? 11 : 12 }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: game.henModelName
                                color: cHen
                                font.pixelSize: root.isNarrow ? 11 : 12
                                font.bold: true
                                elide: Text.ElideMiddle
                                Layout.maximumWidth: parent.width * 0.55
                            }
                        }
                    }
                }

                // 4. Playback Controls Card
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: root.isNarrow ? 106 : 120
                    color: cCardBg
                    radius: 10
                    border.color: "#343b4b"
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: root.cardPadding
                        spacing: root.isNarrow ? 6 : 8

                        Text {
                            text: qsTr("PLAYBACK CONTROLS")
                            color: cTextMuted
                            font.pixelSize: root.isNarrow ? 10 : 11
                            font.bold: true
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: root.isNarrow ? 6 : 8

                            // Play / Pause Button
                            Rectangle {
                                Layout.fillWidth: true
                                height: root.isNarrow ? 32 : 34
                                radius: 6
                                color: game.isPaused ? "#2a4a35" : (pauseMouse.containsMouse ? "#3b4457" : "#313848")
                                border.color: game.isPaused ? cSelect : "#485268"
                                border.width: 1

                                Text {
                                    anchors.centerIn: parent
                                    text: game.isPaused ? (root.isVertical ? qsTr("Play") : qsTr("Play (Space)")) : (root.isVertical ? qsTr("Pause") : qsTr("Pause (Space)"))
                                    color: game.isPaused ? cSelect : cTextMain
                                    font.pixelSize: root.isNarrow ? 11 : 12
                                    font.bold: true
                                }
                                MouseArea {
                                    id: pauseMouse
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.togglePause()
                                }
                            }

                            // Step Once Button
                            Rectangle {
                                Layout.preferredWidth: root.isNarrow ? 56 : 70
                                height: root.isNarrow ? 32 : 34
                                radius: 6
                                color: stepMouse.containsMouse ? "#3b4457" : "#313848"
                                border.color: "#485268"
                                border.width: 1

                                Text {
                                    anchors.centerIn: parent
                                    text: root.isVertical ? qsTr("Step") : qsTr("Step (S)")
                                    color: cTextMain
                                    font.pixelSize: root.isNarrow ? 11 : 12
                                }
                                MouseArea {
                                    id: stepMouse
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.stepOnce()
                                }
                            }

                            // Reset Button
                            Rectangle {
                                Layout.preferredWidth: root.isNarrow ? 56 : 70
                                height: root.isNarrow ? 32 : 34
                                radius: 6
                                color: resetMouse.containsMouse ? "#4a2828" : "#382323"
                                border.color: cFox
                                border.width: 1

                                Text {
                                    anchors.centerIn: parent
                                    text: root.isVertical ? qsTr("Reset") : qsTr("Reset (R)")
                                    color: "#ffa0a0"
                                    font.pixelSize: root.isNarrow ? 11 : 12
                                    font.bold: true
                                }
                                MouseArea {
                                    id: resetMouse
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.resetGame()
                                }
                            }
                        }

                        // Speed Control (FPS)
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: qsTr("AI Speed:")
                                color: cTextMuted
                                font.pixelSize: root.isNarrow ? 11 : 12
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                width: root.isNarrow ? 24 : 26
                                height: width
                                radius: 4
                                color: "#313848"
                                Text { anchors.centerIn: parent; text: "-"; color: cTextMain; font.bold: true; font.pixelSize: 13 }
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.speedFps = Math.max(1, game.speedFps - 1)
                                }
                            }
                            Text {
                                text: game.speedFps + " FPS"
                                color: cTextMain
                                font.pixelSize: root.isNarrow ? 11 : 12
                                font.bold: true
                            }
                            Rectangle {
                                width: root.isNarrow ? 24 : 26
                                height: width
                                radius: 4
                                color: "#313848"
                                Text { anchors.centerIn: parent; text: "+"; color: cTextMain; font.bold: true; font.pixelSize: 13 }
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: game.speedFps = Math.min(20, game.speedFps + 1)
                                }
                            }
                        }
                    }
                }

                Item {
                    Layout.fillHeight: !root.isVertical
                    visible: !root.isVertical
                }

                // 5. Game Over Banner
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: root.isNarrow ? 66 : 70
                    visible: game.gameOver
                    color: game.winner === 1 ? "#2e3b2e" : (game.winner === -1 ? "#3b2222" : "#2f3543")
                    radius: 8
                    border.color: game.winner === 1 ? cHen : (game.winner === -1 ? cFox : cTextMuted)
                    border.width: 1.5

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 6
                        spacing: 2

                        Text {
                            text: game.winnerString
                            color: game.winner === 1 ? cHen : (game.winner === -1 ? cFox : cTextMain)
                            font.pixelSize: root.isNarrow ? 14 : 16
                            font.bold: true
                            Layout.alignment: Qt.AlignHCenter
                        }

                        Text {
                            text: qsTr("Reason: ") + game.terminationReason
                            color: cTextMuted
                            font.pixelSize: 10
                            elide: Text.ElideRight
                            Layout.alignment: Qt.AlignHCenter
                        }

                        Rectangle {
                            width: playAgainText.implicitWidth + 14
                            height: 20
                            radius: 4
                            color: playAgainMouse.containsMouse ? "#3b4457" : "#202530"
                            Layout.alignment: Qt.AlignHCenter
                            border.color: "#546078"
                            border.width: 1

                            Text {
                                id: playAgainText
                                anchors.centerIn: parent
                                text: qsTr("New Game")
                                color: cTextMain
                                font.pixelSize: 11
                                font.bold: true
                            }
                            MouseArea {
                                id: playAgainMouse
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: game.resetGame()
                            }
                        }
                    }
                }
            }
        }
    }
}

// Floating Scroll Indicator for Mobile
Rectangle {
    id: scrollIndicator
    anchors.right: parent.right
    anchors.rightMargin: 3
    y: mainFlickable.y + mainFlickable.visibleArea.yPosition * mainFlickable.height
    width: 4
    height: Math.max(24, mainFlickable.visibleArea.heightRatio * mainFlickable.height)
    radius: 2
    color: "#50a0ff"
    opacity: (mainFlickable.moving || mainFlickable.dragging) ? 0.85 : (mainFlickable.interactive ? 0.35 : 0.0)
    visible: root.isVertical && (mainFlickable.contentHeight > mainFlickable.height)

    Behavior on opacity {
        NumberAnimation { duration: 250 }
    }
}
}


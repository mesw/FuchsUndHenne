cmake_minimum_required(VERSION 3.16)

project(QtFuchsUndHenne VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Quick)

qt_standard_project_setup(REQUIRES 6.8)

qt_add_executable(appQtFuchsUndHenne
    main.cpp
    bitboard.hpp
    inference.hpp
    gameengine.hpp
    gameengine.cpp
)

qt_add_qml_module(appQtFuchsUndHenne
    URI QtFuchsUndHenne
    VERSION 1.0
    QML_FILES
        Main.qml
    SOURCES
        gameengine.hpp
        gameengine.cpp
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appQtFuchsUndHenne PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appQtFuchsUndHenne
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE FALSE
)

target_link_libraries(appQtFuchsUndHenne
    PRIVATE Qt6::Quick
)

# Copy trained AI model weights next to app executable
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../exported_models")
if(NOT EXISTS "${MODEL_DIR}/fox_actor.bin")
    set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(EXISTS "${MODEL_DIR}/fox_actor.bin")
    add_custom_command(TARGET appQtFuchsUndHenne POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_DIR}/fox_actor.bin"
            "$<TARGET_FILE_DIR:appQtFuchsUndHenne>/fox_actor.bin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_DIR}/hens_actor.bin"
            "$<TARGET_FILE_DIR:appQtFuchsUndHenne>/hens_actor.bin"
        COMMENT "Copying AI neural network binary weights next to app executable from ${MODEL_DIR}"
    )

    if(EMSCRIPTEN)
        target_link_options(appQtFuchsUndHenne PRIVATE
            "--preload-file=${MODEL_DIR}/fox_actor.bin@/fox_actor.bin"
            "--preload-file=${MODEL_DIR}/hens_actor.bin@/hens_actor.bin"
        )
    endif()
endif()


# ==============================================================================
# C++ Bitboard & Engine Tests Target (Console executable)
# ==============================================================================
add_executable(test_bitboard
    test_bitboard.cpp
    bitboard.hpp
    inference.hpp
)

set_target_properties(test_bitboard PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    WIN32_EXECUTABLE FALSE
)

# Copy trained AI model weights next to the test binary if present
if(EXISTS "${MODEL_DIR}/fox_actor.bin")
    add_custom_command(TARGET test_bitboard POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_DIR}/fox_actor.bin"
            "$<TARGET_FILE_DIR:test_bitboard>/fox_actor.bin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_DIR}/hens_actor.bin"
            "$<TARGET_FILE_DIR:test_bitboard>/hens_actor.bin"
        COMMENT "Copying AI neural network binary weights next to test_bitboard executable"
    )
endif()

enable_testing()
add_test(NAME test_bitboard COMMAND test_bitboard)

include(GNUInstallDirs)
install(TARGETS appQtFuchsUndHenne
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


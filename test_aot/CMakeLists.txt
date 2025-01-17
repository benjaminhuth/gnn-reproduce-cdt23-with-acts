cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

find_package(Torch REQUIRED)

add_executable(aoti_example inference.cpp model.so)

add_custom_command(
  OUTPUT model.so
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/gnn.so ${CMAKE_CURRENT_BINARY_DIR}/model.so
  DEPENDS gnn.so
)

target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)

cmake_minimum_required(VERSION 2.8)
project(rephotografing)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "D:\\OpenCV\\opencv\\build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV REQUIRED core highgui imgproc xfeatures2d)


add_executable(rephotografing_cv src/Main.cpp src/Main.h src/ModelRegistration.cpp src/ModelRegistration.h src/PnPProblem.cpp src/PnPProblem.h src/RobustMatcher.cpp src/RobustMatcher.h src/Utils.cpp src/Utils.h src/CameraCalibrator.cpp src/CameraCalibrator.h src/Line.cpp src/Line.h src/MSAC.cpp src/MSAC.h src/errorNIETO.cpp src/errorNIETO.h src/lmmin.cpp src/lmmin.h)

# add libs you need
set(OpenCV_LIBS
        opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# linking
target_link_libraries(rephotografing_cv ${OpenCV_LIBS})
target_link_libraries(rephotografing_cv ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBRARIES} opencv_xfeatures2d)

link_directories(/opt/ros/groovy/lib)

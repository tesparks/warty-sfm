cmake_minimum_required(VERSION 2.8)
project( AKAZE_match )

SET(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc)
find_package( OpenCV 3.0 REQUIRED)

MESSAGE("${OpenCV_INCLUDE_DIRS}")

MESSAGE("${OpenCV_LIB_DIR}")

add_executable( AKAZE_match AKAZE_match.cpp )

target_link_libraries( AKAZE_match ${OpenCV_LIBS} )


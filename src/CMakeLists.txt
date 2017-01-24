#FIND_PACKAGE( OpenCV 2.4.11 )
FIND_PACKAGE( PCL 1.7 )

# OpenCV
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${OpenCV_LIBRARY_DIRS} )

# PCL
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (points_grouping main.cpp)
TARGET_LINK_LIBRARIES(points_grouping
    ${OpenCV_LIBS} ${PCL_LIBRARIES})

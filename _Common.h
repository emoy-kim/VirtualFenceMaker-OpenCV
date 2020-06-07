#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

#include "ProjectPath.h"

using uchar = unsigned char;
using uint = unsigned int;

constexpr uint OPENGL_COLOR_BUFFER_BIT = 0x00004000u;
constexpr uint OPENGL_DEPTH_BUFFER_BIT = 0x00000100u;
constexpr uint OPENGL_STENCIL_BUFFER_BIT = 0x00000400u;

inline cv::Scalar RED_COLOR = cv::Scalar(0, 0, 255);
inline cv::Scalar CYAN_COLOR = cv::Scalar(255, 255, 0);
inline cv::Scalar CERULEAN_COLOR = cv::Scalar(211, 64, 2); 
inline cv::Scalar GOLDENROD_COLOR = cv::Scalar(103, 214, 252);
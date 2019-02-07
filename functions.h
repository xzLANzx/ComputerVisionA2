//
// Created by ZelanXiao on 2019/2/7.
//

#ifndef A2_FUNCTIONS_H
#define A2_FUNCTIONS_H

#include "functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace cv;
using namespace std;

void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y);

#endif //A2_FUNCTIONS_H

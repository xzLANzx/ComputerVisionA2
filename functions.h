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

void getNormCornerStrengthMatrix(const Mat &grad_x, const Mat &grad_y, Mat &c_H_norm);

void localMaxSuppression(const Mat &src, Mat &dst, int size, float threshold);

void markKeyPoints(const Mat &img, Mat &dst, const Mat &c_H_norm, float threshold);

#endif //A2_FUNCTIONS_H

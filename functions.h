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

#define PI 3.14159265

void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y);

void getNormCornerStrengthMatrix(const Mat &grad_x, const Mat &grad_y, Mat &c_H_norm);

void localMaxSuppression(const Mat &src, Mat &dst, int size, float threshold);

void getKeyPoints(vector<KeyPoint> &kpt_vec, const Mat &c_H, float threshold);

void getKeypointsOrientations(const Mat &img_orig, int ksize, const vector<KeyPoint> &kpt_vec, vector<KeyPoint> &orient_kpt_vec, int orient_wd_size);

#endif //A2_FUNCTIONS_H

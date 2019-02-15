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

void load(int mode, Mat &img_orig1, Mat &img_orig2);

void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y);

void getNormCornerStrengthMatrix(const Mat &grad_x, const Mat &grad_y, Mat &c_H_norm);

void localMaxSuppression(const Mat &src, Mat &dst, int size, float threshold);

vector<KeyPoint> ANMS(const vector<KeyPoint> &kpt_vec, int limit);

void get2DGaussianFilter(Mat &gauss_coeff, int size, int ktype);

float getFeatureDistance(vector<float> f1, vector<float> f2);

vector<KeyPoint> getKeyPoints(const Mat &c_H, float threshold);

vector<KeyPoint> getOrientedKeyPoints(const Mat &img_orig, const vector<KeyPoint> &kpt_vec, int wd_size);

vector<vector<float>> getFeatureDescriptorsList(const Mat &img, const vector<KeyPoint> &okpt_vec);

vector<vector<float>> getAllKeyPointsFeatureDescriptors(const Mat &img_orig, vector<KeyPoint> &orient_kpt_vec, float threshold, int wd_size);

vector<DMatch> findMatchKeyPoints(const vector<vector<float>> &f_list1, const vector<vector<float>> &f_list2);

#endif //A2_FUNCTIONS_H

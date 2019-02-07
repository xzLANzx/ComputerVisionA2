//
// Created by ZelanXiao on 2019/2/7.
//
#include "functions.h"

void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y) {
    Mat img_gray;
    //convert to gray scale
    cvtColor(img, img_gray, COLOR_RGB2GRAY);

    //default settings
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32FC1;

    //Use Sobel to calculate the derivatives from an image.
    //Use Scharr to calculate a more accurate derivative for a kernel of size 3 * 3

    //gradient X
    Sobel(img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    //gradient Y
    Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
}


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

void getNormCornerStrengthMatrix(const Mat &grad_x, const Mat &grad_y, Mat &c_H_norm){
    int rows = grad_x.rows;
    int cols = grad_x.cols;

    //compute Ix^2, IxIy, Iy^2
    Mat Ix2, IxIy, Iy2;
    pow(grad_x, 2.0, Ix2);
    multiply(grad_x, grad_y, IxIy);
    pow(grad_y, 2.0, Iy2);

    //compute harris matrix
    Mat h_Ix2, h_IxIy, h_Iy2;
    GaussianBlur(Ix2, h_Ix2, Size(5, 5), 0, 0);
    GaussianBlur(IxIy, h_IxIy, Size(5, 5), 0, 0);
    GaussianBlur(Iy2, h_Iy2, Size(5, 5), 0, 0);

    //compute corner strength matrix
    Mat c_H = Mat(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            float h0 = h_Ix2.at<float>(i, j);
            float h1 = h_IxIy.at<float>(i, j);
            float h2 = h_IxIy.at<float>(i, j);
            float h3 = h_Iy2.at<float>(i, j);

            float h_mat_data[4] = {h0, h1, h2, h3};
            Mat h_mat = Mat(2, 2, CV_32F, h_mat_data);

            float det = determinant(h_mat);
            float tra = trace(h_mat).val[0];
            float h_operator = det / tra;

            c_H.at<float>(i, j) = h_operator;
        }
    }

    //normalize the corner strength matrix to (0-255)
    normalize( c_H, c_H_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
}

//size has to be odd
void localMaxSuppression(const Mat &src, Mat &dst, int size, float threshold){
    int rows = src.rows;
    int cols = src.cols;

    Mat result = src.clone();

    int shift = (size - 1) / 2;
    int top, left, right, bottom;

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){

            if(src.at<float>(i, j) >  threshold){   //is a key point

                //check if it is the local maximum in the window
                left = ((j - shift) < 0) ? 0 : j - shift;
                right = ((j + shift) > (cols - 1) ) ? (cols - 1) : j + shift;
                top = ((i - shift) < 0) ? 0 : i - shift;
                bottom = ((i + shift) > (rows - 1)) ? (rows - 1) : i + shift;

                float max = src.at<float>(i, j);
                bool is_max = true;

                for(int m = top; m <= bottom; ++m){
                    for(int n = left; n <= right; ++n){
                        if(src.at<float>(m, n) > max){

                            //suppress dst.at<float>(i, j)
                            result.at<float>(i, j) = 0;
                            is_max = false;
                            break;
                        }
                    }
                    if(is_max == false) break;
                }
            }

        }
    }

    dst = result;
}

void markKeyPoints(const Mat &img, Mat &dst, const Mat &c_H_norm, float threshold){

    vector<KeyPoint> vec;
    int rows = img.rows;
    int cols = img.cols;

    //suppression
    Mat c_H_norm_sup = c_H_norm.clone();
    localMaxSuppression(c_H_norm, c_H_norm_sup, 5, threshold);

    //push key points into vec
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            float c = c_H_norm_sup.at<float>(i, j);
            if (c > threshold) {

                KeyPoint kpt(j, i, 3);
                kpt.response = c;
                vec.push_back(kpt);
            }
        }
    }

    //draw
    drawKeypoints(img, vec, dst);
}

void getOrientationsAroundKeypoint(const Mat &src, int kpt_i, int kpt_j, int wd_size){

}
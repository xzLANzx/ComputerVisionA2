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

void getNormCornerStrengthMatrix(const Mat &grad_x, const Mat &grad_y, Mat &c_H_norm) {
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
    normalize(c_H, c_H_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
}

/**
 *
 * @param src   original corner strength matrix
 * @param dst   return a suppressed corner strength matrix at dst
 * @param size  supression window size
 * @param threshold     threshold to determine whether it is a key point
 */
void localMaxSuppression(const Mat &src, Mat &dst, int size, float threshold) {
    int rows = src.rows;
    int cols = src.cols;

    Mat result = src.clone();

    int shift = (size - 1) / 2;
    int top, left, right, bottom;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            if (src.at<float>(i, j) > threshold) {   //is a key point

                //check if it is the local maximum in the window
                left = ((j - shift) < 0) ? 0 : j - shift;
                right = ((j + shift) > (cols - 1)) ? (cols - 1) : j + shift;
                top = ((i - shift) < 0) ? 0 : i - shift;
                bottom = ((i + shift) > (rows - 1)) ? (rows - 1) : i + shift;

                float max = src.at<float>(i, j);
                bool is_max = true;

                for (int m = top; m <= bottom; ++m) {
                    for (int n = left; n <= right; ++n) {
                        if (src.at<float>(m, n) > max) {

                            //suppress dst.at<float>(i, j)
                            result.at<float>(i, j) = 0;
                            is_max = false;
                            break;
                        }
                    }
                    if (is_max == false) break;
                }
            }
        }
    }

    dst = result;
}

void getKeyPoints(vector<KeyPoint> &kpt_vec, const Mat &c_H, float threshold) {
    int rows = c_H.rows;
    int cols = c_H.cols;

    //push key points into vec
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            float c = c_H.at<float>(i, j);
            if (c > threshold) {

                KeyPoint kpt(j, i, 3);
                kpt.response = c;
                kpt_vec.push_back(kpt);
            }
        }
    }
}


void getKeypointsOrientations(const Mat &img_orig, int ksize, const vector<KeyPoint> &kpt_vec,
                              vector<KeyPoint> &orient_kpt_vec, int orient_wd_size) {
    //convert to gray scale
    Mat img_gray;
    cvtColor(img_orig, img_gray, COLOR_RGB2GRAY);

    //gaussian blur the image
    Mat img_gray_gauss;
    GaussianBlur(img_gray, img_gray_gauss, Size(ksize, ksize), 0, 0);

    vector<float> orient_hist;
    orient_hist.assign(36, 0);
    int kpt_count = kpt_vec.size();
    int rows = img_orig.rows;
    int cols = img_orig.cols;
    int left, right, top, bottom;


    for (int k = 0; k < kpt_count; ++k) {
        KeyPoint kpt = kpt_vec.at(k);
        int i = kpt.pt.y;
        int j = kpt.pt.x;

        int shift = (orient_wd_size - 1) / 2;
        left = ((j - shift) < 0) ? 0 : j - shift;
        right = ((j + shift) > (cols - 1)) ? (cols - 1) : j + shift;
        top = ((i - shift) < 0) ? 0 : i - shift;
        bottom = ((i + shift) > (rows - 1)) ? (rows - 1) : i + shift;

        //loop through all the pixels in window
        for (int m = (top + 1); m < bottom; ++m) {
            for (int n = (left + 1); n < right; ++n) {

                //calculate magnitude
                float a, b, c, d;
                a = img_gray_gauss.at<uchar>(m, n + 1);
                b = img_gray_gauss.at<uchar>(m, n - 1);
                c = img_gray_gauss.at<uchar>(m + 1, n);
                d = img_gray_gauss.at<uchar>(m - 1, n);

                float kpt_magn = sqrt(pow(a - b, 2) + pow(c - d, 2));

                //calculate theta
                float delta_y = c - d;
                float delta_x = a - b;
                double theta = atan2(delta_y, delta_x) * 180 / PI;
                if (theta < 0) theta += 360;

                int bucket_idx = (int) (theta / 10);
                orient_hist[bucket_idx] += kpt_magn;
            }
        }

        //normalize the histogram
        vector<float> orient_hist_norm;
        normalize(orient_hist, orient_hist_norm, 0, 100, NORM_MINMAX, CV_32FC1, Mat());

        //find all the prominent orientations with strength > 80
        for (int p = 0; p < orient_hist_norm.size(); ++p) {
            if (orient_hist_norm[p] >= 80) {
                int angle = p * 10;
                KeyPoint orient_kpt = KeyPoint(j, i, (kpt.response / 4), angle, kpt.response);
                orient_kpt_vec.push_back(orient_kpt);
            }
        }
    }
}








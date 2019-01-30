#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace cv;
using namespace std;


void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y) {
    Mat img_gray;
    //convert to gray scale
    cvtColor(img, img_gray, COLOR_RGB2GRAY);

    //default settings
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32FC1;

    //gradient X
    Sobel(img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    //gradient Y
    Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
}


Mat getHarrisMatrix(const Mat &gau_mask, const Mat &grad_x, const Mat &grad_y, int u, int v) {

    float Ix = grad_x.at<float>(u, v);
    float Iy = grad_y.at<float>(u, v);

    float coef_data[4] = {Ix * Ix, Ix * Iy, Ix * Iy, Iy * Iy};
    Mat coef_mat = Mat(2, 2, CV_32F, coef_data);

    Mat h_mat = Mat(2, 2, CV_32F, Scalar(0.0f));
    for (int i = 0; i < gau_mask.rows; ++i) {
        for (int j = 0; j < gau_mask.cols; ++j) {
            h_mat = h_mat + gau_mask.at<float>(i, j) * coef_mat;
        }
    }
    return h_mat;
}


float getHarrisOperator(const Mat &h) {
    float det = determinant(h);
    float tra = trace(h).val[0];
    if (tra == 0) return 0;
    else return det / tra;
}


void getCornerStrengthMatrix(const Mat &img, const Mat &gau_mat, Mat &corner_mat) {
    Mat grad_x, grad_y;
    getGradient(img, grad_x, grad_y);


    corner_mat = Mat(img.rows, img.cols, CV_32F);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Mat h_mat = getHarrisMatrix(gau_mat, grad_x, grad_y, i, j);
            float c = getHarrisOperator(h_mat);
            corner_mat.at<float>(i, j) = c;
        }
    }
}


int main(int argc, char *argv[]) {

    //load image into Matrix
    string img_name(argv[1]);    //argv[1] names of the image
    string f_name = "../image_set/" + img_name;
    Mat img_orig = imread(f_name, IMREAD_COLOR);


    float gaussian_mask_data[25] = {
            1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1
    };
    Mat w = Mat(Size(5, 5), CV_32F, gaussian_mask_data);
    w = w / (273.0);


    Mat corner_strength_mat;
    getCornerStrengthMatrix(img_orig, w, corner_strength_mat);

    corner_strength_mat = 1000000 * corner_strength_mat;
    float threshold = 50000.0f;
    // Drawing a circle around corners
    for (int i = 0; i < img_orig.rows; ++i) {
        for (int j = 0; j < img_orig.cols; ++j) {
            if (corner_strength_mat.at<float>(i, j) > threshold) {
                circle(img_orig, Point(j, i), 3, Scalar(255), 2, 8, 0);
            }
        }
    }


//    cout<<grad_x<<endl;
    imshow("Orig", img_orig);
//    imshow("Gray", img_gray);

    waitKey(0);

    return 0;
}
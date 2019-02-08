#include "functions.h"

int main(int argc, char *argv[]) {
    //default settings
    float threshold = 80.0f;


    //load image into Matrix
    string img_name(argv[1]);    //argv[1] names of the image
    string f_name = "../image_set/" + img_name;
    Mat img_orig = imread(f_name, IMREAD_COLOR);


    //compute Ix, Iy
    Mat grad_x, grad_y;
    getGradient(img_orig, grad_x, grad_y);


    //compute a normalized CornerStrengthMatrix
    Mat c_H_norm;
    getNormCornerStrengthMatrix(grad_x, grad_y, c_H_norm);


    //mark detected key points on original image
    markKeyPoints(img_orig, img_orig, c_H_norm, threshold);
    imshow("Result", img_orig);


    waitKey(0);
    return 0;
}
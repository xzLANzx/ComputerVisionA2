#include "functions.h"

int main(int argc, char *argv[]) {
    //default settings
    float threshold = 80.0f;
    int ksize = 91; //orient_wd_size = 21
    float sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    int orient_wd_size = 1.5 * sigma;       //need to make sure orient_wd_size is odd


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

    //suppress the corner strength matrix
    Mat c_H_suppressed;
    localMaxSuppression(c_H_norm, c_H_suppressed, 5, threshold);


    //get key points
    vector<KeyPoint> kpt_vec;
    getKeyPoints(kpt_vec, c_H_suppressed, threshold);


    //draw key points
    Mat img_kpts;
    drawKeypoints(img_orig, kpt_vec, img_kpts);
    imshow("Key Points", img_kpts);


    //draw oriented key points
    Mat img_orient_kpts;                //image with oriented keypoints
    vector<KeyPoint> orient_kpt_vec;
    getKeypointsOrientations(img_orig, ksize, kpt_vec, orient_kpt_vec, orient_wd_size);
    drawKeypoints(img_orig, orient_kpt_vec, img_orient_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Oriented Key Points", img_orient_kpts);


    cout<<kpt_vec.size()<<endl;
    cout<<orient_kpt_vec.size()<<endl;


    waitKey(0);
    return 0;
}
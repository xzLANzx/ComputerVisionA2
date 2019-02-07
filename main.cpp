#include "functions.h"

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
    Mat gaussianFilter = Mat(Size(5, 5), CV_32F, gaussian_mask_data);
    gaussianFilter = gaussianFilter / (273.0);

    //compute Ix, Iy
    Mat grad_x, grad_y;
    getGradient(img_orig, grad_x, grad_y);

    //compute Ix^2, IxIy, Iy^2
    Mat Ix2, IxIy, Iy2;
    pow(grad_x, 2.0, Ix2);
    multiply(grad_x, grad_y, IxIy);
    pow(grad_y, 2.0, Iy2);

    //compute harris matrix
    Mat h_Ix2, h_IxIy, h_Iy2;
    filter2D(Ix2, h_Ix2, Ix2.depth(), gaussianFilter);
    filter2D(IxIy, h_IxIy, IxIy.depth(), gaussianFilter);
    filter2D(Iy2, h_Iy2, Iy2.depth(), gaussianFilter);

    //compute corner strength matrix
    Mat c_H = Mat(img_orig.rows, img_orig.cols, CV_32F);
    for (int i = 0; i < img_orig.rows; ++i) {
        for (int j = 0; j < img_orig.cols; ++j) {
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

    Mat c_H_norm;
    //normalizing result from 0 to 255
    normalize( c_H, c_H_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );

    Mat img_copy = img_orig.clone();
    //mark points with corner strength greater than the threshold
    float threshold = 62.5f;
    // Drawing a circle around corners
    vector<KeyPoint> vec;
    for (int i = 0; i < img_orig.rows; ++i) {
        for (int j = 0; j < img_orig.cols; ++j) {

            float c = c_H_norm.at<float>(i, j);


            if (c > threshold) {
                //circle(img_orig, Point(j, i), 2, Scalar(255, 0, 255), 1, 8, 0);

                KeyPoint kpt(j, i, 3);
                kpt.response = c;
                vec.push_back(kpt);

                //check if it is local maximum in 5*5 window
                int top, left, right, bottom;
                left = ((j - 2) < 0) ? 0 : i - 2;
                right = ((j + 2) > (img_orig.cols - 1) ) ? (img_orig.cols - 1) : j + 2;
                top = ((i - 2) < 0) ? 0 : i - 2;
                bottom = ((i + 2) > (img_orig.rows - 1)) ? (img_orig.rows - 1) : i + 2;

                bool is_max = true;
                for(int m = top; m <= bottom; ++m){
                    for(int n = left; n <= right; ++n){
                        if(c_H.at<float>(m, n) >  c){
                            is_max = false;
                            goto theEnd;
                        }
                    }
                }
                theEnd:;

                if(is_max){
                    //circle(img_copy, Point(j, i), 2, Scalar(255, 255, 0), 1, 8, 0);

//                    KeyPoint kpt(j, i, 3);
//                    kpt.response = c;
//                    vec.push_back(kpt);

                }
            }

        }
    }
    drawKeypoints(img_orig, vec, img_orig);



    Mat comparison;
    hconcat(img_orig, img_copy, comparison);
    imshow("Result", comparison);

    Mat gauFilter = getGaussianKernel(5, 1);
    cout<<gauFilter<<endl;





    waitKey(0);

    return 0;
}
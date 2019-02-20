//
// Created by ZelanXiao on 2019/2/7.
//
#include "functions.h"

void load(int mode, Mat &img_orig1, Mat &img_orig2) {
    //Default mode 1: yosemite contrast invariance
    //mode 2: yosemite rotation invariance
    //mode 3: killing turkey

    string f_name1 = "../image_set/img1.ppm";
    string f_name2 = "../image_set/img2.ppm";

    if (mode == 1) {

        f_name1 = "../image_set/Yosemite1.jpg";
        f_name2 = "../image_set/Yosemite2.jpg";

        img_orig1 = imread(f_name1, IMREAD_COLOR);
        img_orig2 = imread(f_name2, IMREAD_COLOR);

    } else if (mode == 2) {

        f_name1 = "../image_set/Yosemite1.jpg";
        f_name2 = "../image_set/Yosemite1_rotation.jpg";

        img_orig1 = imread(f_name1, IMREAD_COLOR);
        img_orig2 = imread(f_name2, IMREAD_COLOR);

    } else if (mode == 3) {


        f_name1 = "../image_set/img1.ppm";
        f_name2 = "../image_set/img2.ppm";

        img_orig1 = imread(f_name1, IMREAD_COLOR);
        img_orig2 = imread(f_name2, IMREAD_COLOR);

//        // rotation factor
//        float theta = 15;
//        theta = theta * PI / 180;
//        float si = sin(theta);
//        float co = cos(theta);
//
//        // rotation parameter
//        float rot[2][2] = { {co, -1*si},
//                            {si, co} };
//
//        Mat rotImg = Mat::zeros(img_orig2.rows, img_orig2.cols, img_orig2.type());
//
//        for (int y = 0; y < img_orig2.rows; y++) {
//            for (int x = 0; x < img_orig2.cols; x++) {
//                int x_new = rot[0][0] * x + rot[0][1] * y;
//                int y_new = rot[1][0] * x + rot[1][1] * y;
//
//                // make sure it's not going out of boundary
//                if (x_new >= 0 && x_new < img_orig2.cols && y_new >=0 && y_new < img_orig2.rows) {
//                    rotImg.at<Vec3b>(y_new, x_new) = img_orig2.at<Vec3b>(y, x);
//                }
//            }
//        }
//
//        img_orig2.release();
//        img_orig2 = rotImg;
    } else {
        cout << "Please select a valid option. " << endl;
    }


}

void getGradient(const Mat &img, Mat &grad_x, Mat &grad_y) {
    Mat img_gray;
    //convert to gray scale
    cvtColor(img, img_gray, COLOR_RGB2GRAY);

    //default settings
    int scale = 1;
    int delta = 0;
    auto ddepth = CV_32FC1;

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

            double det = determinant(h_mat);
            double tra = trace(h_mat).val[0];
            double h_operator = det / tra;

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
                    if (!is_max) break;
                }
            }
        }
    }

    dst = result;
}

vector<KeyPoint> ANMS(const vector<KeyPoint> &kpt_vec, int limit) {

    vector<KeyPoint> spatial_kpt_vec;
    size_t kpt_count = kpt_vec.size();
    vector<double> radius_vec(kpt_count, INFINITY);         //all key points' initial radius

    for (int i = 0; i < kpt_count; ++i) {
        KeyPoint x_i = kpt_vec[i];
        for (int j = 0; j < kpt_count; ++j) {
            if (j != i) {
                KeyPoint x_j = kpt_vec[j];
                if ((0.9 * x_j.response) > x_i.response) {
                    radius_vec[i] = norm(x_i.pt - x_j.pt);
                }
            }
        }
    }

    //sort {radius, index} in descending order
    vector<int> idx_vec(radius_vec.size());
    for (int i = 0; i < radius_vec.size(); ++i) {
        idx_vec[i] = i;
    }

    sort(idx_vec.begin(), idx_vec.end(),
         [&radius_vec](int i, int j) { return (radius_vec[i] > radius_vec[j]); });

    for (int i = 0; i < kpt_count; ++i) {
        int idx = idx_vec[i];
        if (radius_vec[i] > limit)
            spatial_kpt_vec.push_back(kpt_vec[idx]);
        else
            break;
    }

    return spatial_kpt_vec;
}


vector<KeyPoint> getKeyPoints(const Mat &c_H, float threshold) {

    vector<KeyPoint> kpt_vec;

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

    return kpt_vec;
}

vector<KeyPoint> getOrientedKeyPoints(const Mat &img_orig, const vector<KeyPoint> &kpt_vec, int wd_size) {

    //convert to gray scale
    Mat img_gray;
    cvtColor(img_orig, img_gray, COLOR_RGB2GRAY);


    //gaussian blur the image
    Mat img_gray_gauss;
    GaussianBlur(img_gray, img_gray_gauss, Size(5, 5), 0, 0);


    vector<KeyPoint> orient_kpt_vec;
    vector<float> orient_hist;
    orient_hist.assign(36, 0);
    size_t kpt_count = kpt_vec.size();
    int rows = img_orig.rows;
    int cols = img_orig.cols;
    int left, right, top, bottom;


    for (int k = 0; k < kpt_count; ++k) {
        KeyPoint kpt = kpt_vec.at(k);
        int i = kpt.pt.y;
        int j = kpt.pt.x;

        int shift = (wd_size - 1) / 2;
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
    return orient_kpt_vec;
}

void get2DGaussianFilter(Mat &gauss_coeff, int size, int ktype) {
    auto gauss_x = cv::getGaussianKernel(size, 0, ktype);
    auto gauss_y = cv::getGaussianKernel(size, 0, ktype);
    gauss_coeff = gauss_x * gauss_y.t();
}

vector<vector<float>> getFeatureDescriptorsList(const Mat &img, const vector<KeyPoint> &okpt_vec) {

    vector<vector<float>> feature_descriptor_vec;
    int rows = img.rows;
    int cols = img.cols;


    //convert to gray scale
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_RGB2GRAY);


    int left, right, top, bottom;
    int shift = 8;


    for (int k = 0; k < okpt_vec.size(); ++k) {

        KeyPoint okpt = okpt_vec.at(k);
        int i = okpt.pt.y;
        int j = okpt.pt.x;
        float orient = okpt.angle;

        //Generate a 18 by 18 window around the keypoint
        left = j - shift;
        right = j + shift + 1;
        top = i - shift;
        bottom = i + shift + 1;


        //loop through all the pixels in window
        Mat magn_window = Mat(16, 16, CV_32F);
        Mat orient_window = Mat(16, 16, CV_32F);


        float pt_magn, theta;

        for (int m = (top + 1); m < bottom; ++m) {
            for (int n = (left + 1); n < right; ++n) {

                if ((m > 0) && (m < (rows - 1)) && (n > 0) && (n < (cols - 1))) {

                    //Calculate the magnitude
                    float a, b, c, d;
                    a = img_gray.at<uchar>(m, n + 1);
                    b = img_gray.at<uchar>(m, n - 1);
                    c = img_gray.at<uchar>(m + 1, n);
                    d = img_gray.at<uchar>(m - 1, n);

                    pt_magn = sqrt(pow(a - b, 2) + pow(c - d, 2));

                    //calculate theta
                    float delta_y = c - d;
                    float delta_x = a - b;
                    theta = atan2(delta_y, delta_x) * 180 / PI;
                    if (theta < 0) theta += 360;

                } else {

                    pt_magn = 0.0f;
                    theta = -1;

                }

                magn_window.at<float>(m - (top + 1), n - (left + 1)) = pt_magn;
                orient_window.at<float>(m - (top + 1), n - (left + 1)) = theta;

            }
        }




        //copy magnitude to an odd size matrix to do gaussian weighting
        int w_size = 17;
        Mat magn_window_17 = Mat(w_size, w_size, CV_32F);

        //set col 0 all to zeros
        for (int m = 0; m < w_size; ++m) {
            magn_window_17.at<float>(m, 0) = 0.0f;
        }

        //set row 0 all to zeros
        for (int n = 0; n < w_size; ++n) {
            magn_window_17.at<float>(0, n) = 0.0f;
        }

        //copy the 16 by 16 magnitude to it
        for (int m = 1; m < w_size; ++m) {
            for (int n = 1; n < w_size; ++n) {
                magn_window_17.at<float>(m, n) = magn_window.at<float>(m - 1, n - 1);
            }
        }

        //generate 17 by 17 gaussian coefficients
        Mat gauss_filter;
        get2DGaussianFilter(gauss_filter, w_size, CV_32F);

        //weighted the magnitude
        Mat magn_window_17_weighted = Mat(w_size, w_size, CV_32F);
        for (int m = 0; m < w_size; ++m) {
            for (int n = 0; n < w_size; ++n) {
                magn_window_17_weighted.at<float>(m, n) = magn_window_17.at<float>(m, n) * gauss_filter.at<float>(m, n);
            }
        }

        //put into histogram
        //generate an empty feature descriptor
        vector<float> histogram(8, 0.0f);
        vector<vector<float>> hist_list;

        for (int h = 0; h < 16; h++) {
            hist_list.push_back(histogram);
        }

        for (int m = 0; m < 16; ++m) {
            for (int n = 0; n < 16; ++n) {
                float magn = magn_window_17_weighted.at<float>(m + 1, n + 1);
                float angle = orient_window.at<float>(m, n);

                if (angle >= 0) {
                    //put the magnitude into correct histogram
                    int bin_row = m / 4;
                    int bin_col = n / 4;
                    int bin_idx = bin_row * 4 + bin_col;

                    vector<float> hist = hist_list[bin_idx];
                    angle = angle - orient;
                    if (angle < 0) angle = angle + 360;
                    if (angle == 360) angle = 0;

                    int sector_idx = (int) (angle / 45);
                    if (sector_idx >= 8 || sector_idx < 0)
                        cout << "Error!" << sector_idx << ", Angle: " << angle << endl;
                    hist[sector_idx] += magn;
                    hist_list[bin_idx] = hist;
                }
            }
        }

        vector<float> feature_descriptor;
        for (int m = 0; m < hist_list.size(); m++) {
            for (int n = 0; n < hist_list[0].size(); n++) {
                feature_descriptor.push_back(hist_list[m].at(n));
            }
        }

        //normalize 128-d feature descriptor
        vector<float> feature_descriptor_norm;
        normalize(feature_descriptor, feature_descriptor_norm, 1.0, 0.0, NORM_L2);

        //clamp to 0.2
        for (int f = 0; f < feature_descriptor_norm.size(); f++) {
            if (feature_descriptor_norm[f] >= 0.2) feature_descriptor_norm[f] = 0.2;
        }

        //normalize again
        normalize(feature_descriptor_norm, feature_descriptor, 0, 1, NORM_MINMAX);

        feature_descriptor_vec.push_back(feature_descriptor);
    }
    return feature_descriptor_vec;
}

float getFeatureDistance(vector<float> f1, vector<float> f2) {
    float distance = 0;
    for (int i = 0; i < f1.size(); ++i) {
        distance += abs(f1[i] - f2[i]);
    }
    return distance;
}

vector<vector<float>>
getAllKeyPointsFeatureDescriptors(const Mat &img_orig, vector<KeyPoint> &orient_kpt_vec, float threshold, int wd_size) {
    //compute Ix, Iy
    Mat grad_x, grad_y;
    getGradient(img_orig, grad_x, grad_y);

    //compute a normalized CornerStrengthMatrix
    Mat c_H_norm;
    getNormCornerStrengthMatrix(grad_x, grad_y, c_H_norm);

    //local maximum suppression
    Mat c_H_norm_suppresed;
    localMaxSuppression(c_H_norm, c_H_norm_suppresed, 11, threshold);

    //get keypoints
    vector<KeyPoint> kpt_vec = getKeyPoints(c_H_norm_suppresed, threshold);

    //Adaptive Non Maximum Suppression
    vector<KeyPoint> spatial_kpt_vec = ANMS(kpt_vec, 25);
    cout << "Spatial Key Points Count: " << spatial_kpt_vec.size() << endl;

    //get oriented keypoints
    orient_kpt_vec = getOrientedKeyPoints(img_orig, spatial_kpt_vec, wd_size);
    cout << "Oriented Key Points Count: " << orient_kpt_vec.size() << endl << endl;

    //vector of all keypoints' feature descriptors
    vector<vector<float>> feature_descriptor_vec = getFeatureDescriptorsList(img_orig, orient_kpt_vec);



//    //draw oriented key points
//    Mat img_orient_kpts;
//    drawKeypoints(img_orig, orient_kpt_vec, img_orient_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    imshow("Oriented Key Points", img_orient_kpts);

    return feature_descriptor_vec;
}

vector<DMatch> findMatchKeyPoints(const vector<vector<float>> &f_list1, const vector<vector<float>> &f_list2, int mode) {

    vector<DMatch> dMatches;
    size_t kpt_count1 = f_list1.size();
    size_t kpt_count2 = f_list2.size();


    //threshold of the two feature descriptors' distance
    float threshold;
    if(mode == 1 || mode == 2)
        threshold = 12;
    else if(mode == 3)
        threshold = 13;

    float d = INFINITY;

    for (int i = 0; i < kpt_count1; ++i) {

        vector<float> buffer;       //store the possible matches's distance
        vector<int> idx_buff;       //store the corresponding match's index
        for (int j = 0; j < kpt_count2; ++j) {
            d = getFeatureDistance(f_list1[i], f_list2[j]);
            if (d < threshold) {
                buffer.push_back(d);
                idx_buff.push_back(j);
            }
        }

        //ratio test
        //find min distance
        float min = INFINITY;
        int min_idx = -1;
        for (int k = 0; k < buffer.size(); ++k) {
            if (buffer[k] < min) {
                min = buffer[k];
                min_idx = idx_buff[k];

                //set buffer[k] to a large number
                buffer[k] = INFINITY;
            }
        }

        //find 2nd min distance
        float sec_min = INFINITY;
        float sec_min_idx = -1;
        for (int k = 0; k < buffer.size(); ++k) {
            if (buffer[k] < sec_min) {
                sec_min = buffer[k];
                sec_min_idx = idx_buff[k];
            }
        }

        //compute ratio
        float ratio = INFINITY;
        if ((min_idx >= 0) && (sec_min_idx >= 0)) {
            if (sec_min > 0) ratio = min / sec_min;
        }

        //keep the max and it's corresponding index
        DMatch dMatch;

        float ratio_threshold = 0;
        if(mode == 1 || mode == 2)
            ratio_threshold = 0.78;
        else if(mode == 3)
            ratio_threshold = 0.57;

        if (ratio < ratio_threshold) {
            dMatch = DMatch(i, min_idx, 3);
            dMatches.push_back(dMatch);
        }
    }

//    d = 0;
//    float min_d = INFINITY;
//    float max_d = 0;
//    for (int i = 0; i < kpt_count1; ++i) {
//        for (int j = 0; j < kpt_count2; ++j) {
//            d = getFeatureDistance(f_list1[i], f_list2[j]);
//            if(d<min_d) min_d = d;
//            if(d>max_d) max_d = d;
//        }
//    }
//
//    cout<<"min: "<<min_d<<endl;
//    cout<<"max: "<<max_d<<endl;

    return dMatches;
}

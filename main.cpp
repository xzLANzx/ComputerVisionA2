#include "functions.h"

int main(int argc, char *argv[]) {
    //default settings
//    float threshold = 80.0f;
//    int ksize = 91; //orient_wd_size = 21
//    float sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
//    int orient_wd_size = 1.5 * sigma;       //need to make sure orient_wd_size is odd

    float threshold = 80.0f;
    int ksize = 7; //orient_wd_size = 21
    float sigma = 3;
    int orient_wd_size = 21;       //need to make sure orient_wd_size is odd


    //load image into Matrix
    string img_name(argv[1]);    //argv[1] names of the image
    //string f_name1 = "../image_set/" + img_name;
    //string f_name1 = "../image_set/img1.ppm";
    string f_name1 = "../image_set/Yosemite1.jpg";
    Mat img_orig1 = imread(f_name1, IMREAD_COLOR);

    //string f_name2 = "../image_set/img2.ppm";
    string f_name2 = "../image_set/Yosemite2.jpg";
    Mat img_orig2 = imread(f_name2, IMREAD_COLOR);

    vector<vector<float>> feature_descriptor_list1;
    vector<KeyPoint> kpt_vec1, orient_kpt_vec1;
    getImageFeatureDescriptors(img_orig1, feature_descriptor_list1, kpt_vec1, orient_kpt_vec1, threshold, ksize, orient_wd_size);

    vector<vector<float>> feature_descriptor_list2;
    vector<KeyPoint> kpt_vec2, orient_kpt_vec2;
    getImageFeatureDescriptors(img_orig2, feature_descriptor_list2, kpt_vec2, orient_kpt_vec2, threshold, ksize, orient_wd_size);

//    cout<<feature_descriptor_list.size()<<endl;
//    cout<<feature_descriptor_list2.size()<<endl;

    vector<KeyPoint> match_points_vec1;
    vector<KeyPoint> match_points_vec2;
    vector<DMatch> dMatches;
    findMatchKeyPoints(feature_descriptor_list1, feature_descriptor_list2, orient_kpt_vec1, orient_kpt_vec2, match_points_vec1, match_points_vec2, dMatches);

    cout<<match_points_vec1.size()<<endl;
    cout<<match_points_vec2.size()<<endl;

    Mat outImg;

    drawMatches(img_orig1, orient_kpt_vec1, img_orig2, orient_kpt_vec2, dMatches, outImg);
    imshow("Matches", outImg);

    waitKey(0);
    return 0;
}
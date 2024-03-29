#include "functions.h"

int main(int argc, char *argv[]) {
    //default settings
    float threshold = 70.0f;
    int orient_wd_size = 21;        //need to make sure orient_wd_size is odd

    //load image into Matrix
    int mode =0;

    cout<<"Please choose an option: "<<endl;
    cout<<"1. View Yosemite Matching Result (Contrast Invariance). "<<endl;
    cout<<"2. View Yosemite Matching Result (Rotation Invariance). "<<endl;
    cout<<"3. View Sword Turkey Matching Result. "<<endl;

    cin>>mode;
    while(mode<1 || mode >3){
        cout<<"Please choose a valid option: "<<endl;
        cout<<"1. View Yosemite Matching Result (Contrast Invariance). "<<endl;
        cout<<"2. View Yosemite Matching Result (Rotation Invariance). "<<endl;
        cout<<"3. View Sword Turkey Matching Result. "<<endl;
    }

    Mat img_orig1, img_orig2;
    load(mode, img_orig1, img_orig2);


    //generate feature descriptor vectors for both two images
    vector<KeyPoint> orient_kpt_vec1, orient_kpt_vec2;
    vector<vector<float>> feature_descriptor_list1 = getAllKeyPointsFeatureDescriptors(img_orig1, orient_kpt_vec1,
                                                                                       threshold, orient_wd_size);
    vector<vector<float>> feature_descriptor_list2 = getAllKeyPointsFeatureDescriptors(img_orig2, orient_kpt_vec2,
                                                                                       threshold, orient_wd_size);

    //draw key points
    Mat img_kpts1;
    drawKeypoints(img_orig1, orient_kpt_vec1, img_kpts1);
    imshow("Image 1 Key Points", img_kpts1);

    Mat img_kpts2;
    drawKeypoints(img_orig2, orient_kpt_vec2, img_kpts2);
    imshow("Image 2 Key Points", img_kpts2);

    cout << "img1 key points count: " << feature_descriptor_list1.size() << endl;
    cout << "img2 key points count: " << feature_descriptor_list2.size() << endl;

    vector<DMatch> dMatches = findMatchKeyPoints(feature_descriptor_list1, feature_descriptor_list2, mode);
    cout << "Matches count: " << dMatches.size() << endl;

    //draw matches
    Mat outImg;
    drawMatches(img_orig1, orient_kpt_vec1, img_orig2, orient_kpt_vec2, dMatches, outImg);
    imshow("Matches", outImg);

    waitKey(0);

    return 0;
}
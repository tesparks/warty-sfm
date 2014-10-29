#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 4000.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.7f;   // Nearest neighbor matching ratio

void findCameraMatrix(Mat K_camera) {
	double focal_length_mm  = 24;
	double sensor_width_mm = 36; //full frame
	double sensor_height_mm = 24; //full frame
	unsigned int image_width_px = 2813;
	unsigned int image_height_px = 2813*3/2;
	double principal_point_x_px = image_width_px/2;
	double principal_point_y_px = image_height_px/2;
	
	double focal_length_x_px = image_width_px * focal_length_mm / sensor_width_mm;
	double focal_length_y_px = image_height_px * focal_length_mm / sensor_height_mm;
	
	K_camera = (focal_length_x_px, 0, principal_point_x_px,
										0, focal_length_y_px, principal_point_y_px,
										0, 0, 1);
	
}


void findFundamentalMatrix(vector<Point2f> matched_pts1, vector<Point2f> matched_pts2, Mat img1) {
	std::vector<uchar> status;
	
	const double threshold(4.0*std::max(img1.size().width, img1.size().height));
	Mat fundamental_mat = findFundamentalMat(cv::Mat{matched_pts1, true}, cv::Mat{matched_pts2, true}, status, cv::FM_RANSAC, threshold);
	cout << fundamental_mat << endl;
}


int main(void)
{

    printf("working now\n");
    
    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

    Mat homography;
    FileStorage fs("H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;
    std::cout << homography <<"\n";
    
    
    printf("working now\n");

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    AKAZE akaze;
    akaze(img1, noArray(), kpts1, desc1);
    //OrbFeatureDetector detector(400);
    //detector.detect( img1, kpts1 );
    
    printf("working now\n");
    akaze(img2, noArray(), kpts2, desc2);
    
    
    printf("working now\n");

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    
    printf("working now\n");

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    printf("working now 6\n");

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    printf("working now 7\n");

    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("res2.png", res);
    
    vector<Point2f> matched_pts1, matched_pts2;
    for (size_t i = 0; i < good_matches.size(); i++) {
    		int i1 = good_matches[i].queryIdx;
    		int i2 = good_matches[i].trainIdx;
    		const KeyPoint &kp1 = inliers1[i1];
    		const KeyPoint &kp2 = inliers2[i2];
    		Point2f pt1 = kp1.pt, 
    				pt2 = kp2.pt;
    		matched_pts1.push_back(pt1);
    		matched_pts2.push_back(pt2);
			Scalar color = Scalar((pt2.x - pt1.x) * 2, (pt1.y - pt2.y) * 2, 0);
    		circle(img1, pt1, 5, color, 3, 8, 0);
    }
    findFundamentalMatrix(matched_pts1, matched_pts2, img1);
    imwrite("res3.png", img1);
    
    Mat camera_matrix = Mat_<double>(3,3);
    findCameraMatrix(camera_matrix);
    
    //Mat triangulated;
    //triangulatePoints(camera_matrix, camera_matrix, matched_pts1, matched_pts2, triangulated);
    
    
    printf("working now 8\n");

    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;

    return 0;
}


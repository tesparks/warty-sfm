#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 4000.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.7f;   // Nearest neighbor matching ratio

void findProjectionMatrix(Mat K_1, Mat K_2, Mat F, Mat Projection_Mat_1, Mat Projection_Mat_2) {
	Mat_<double> essential_mat = K_1.t() * F * K_2;
	
	Matx33d delta(0,-1,0,
			1,0,0,
			0,0,1);
			
	SVD svd(essential_mat);
	Mat_<double> r = svd.u * Mat(delta) * svd.vt;
	
	Mat_<double> t = svd.u.col(2);
	//Mat_<double> Rt = Mat(Matx34d(R(0,0), R(0,1), R(0, 2), t(0),
	//						R(1,0), R(1,1), R(1, 2), t(1),
	//						R(2,0), R(2,1), R(2, 2), t(2)));
	//						
	//Mat projection_mat_2 = K_2*Rt;
	//
	//Mat_<double> projection_mat_1 = Mat(Matx34d(1,0,0,0,
	//										 0,1,0,0,
	//										 0,0,1,0));
	//
	//projection_mat_1 = K_1 * projection_mat_1;

	Mat_<double> distCoeffs = Mat(Matx33d(1, 0, 1, 
										  0, 1, 1, 
										  0, 0, 1));

	Size imgSize = Size(2813, 1873);
	Mat R_1 = Mat_<double>(3, 3);
	Mat R_2 = Mat_<double>(3, 3);

	Mat projection_mat_1 = Mat_<double>(3, 4);
	Mat projection_mat_2 = Mat_<double>(3, 4);

	Mat ddm = Mat_<double>(4, 4);

	stereoRectify(K_1, distCoeffs, K_2, distCoeffs, imgSize, r, t, R_1, R_2,
		projection_mat_1, projection_mat_2, ddm, 0, -1, imgSize);

	
	projection_mat_1.copyTo(Projection_Mat_1);
	projection_mat_2.copyTo(Projection_Mat_2);
	
	return;
}



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
	
	K_camera.at<double>(0,0,0) = focal_length_x_px;
	K_camera.at<double>(0,2,0) = principal_point_x_px;
	K_camera.at<double>(1,1,0) = focal_length_y_px;
	K_camera.at<double>(1,2,0) = principal_point_y_px;
	K_camera.at<double>(2,2,0) = 1;
}


Mat findFundamentalMatrix(vector<Point2f> matched_pts1, vector<Point2f> matched_pts2, Mat img1) {
	std::vector<uchar> status;
	
	const double threshold(4.0*std::max(img1.size().width, img1.size().height));
	Mat fundamental_mat = findFundamentalMat(cv::Mat{matched_pts1, true}, cv::Mat{matched_pts2, true}, status, cv::FM_RANSAC, threshold);
	return fundamental_mat;
}


int main(int argc, char* argv[])
{
    
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
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMatrix(matched_pts1, matched_pts2, img1);
    cout << fundamental_matrix << endl;
    imwrite("res3.png", img1);
    
    Mat camera_matrix_1 = Mat_<double>(3,3);
    Mat camera_matrix_2 = Mat_<double>(3,3);
    camera_matrix_1 = Scalar(0);
    camera_matrix_2 = Scalar(0);
    findCameraMatrix(camera_matrix_1);
    findCameraMatrix(camera_matrix_2);
    
    Mat projection_matrix_1 = Mat_<double>(3,4);
    Mat projection_matrix_2 = Mat_<double>(3,4);
  
    
    findProjectionMatrix(camera_matrix_1, camera_matrix_2, fundamental_matrix, projection_matrix_1, projection_matrix_2);
    
    
    Mat triangulated;
    triangulatePoints(projection_matrix_1, projection_matrix_2, matched_pts1, matched_pts2, triangulated);
    
    cout << triangulated.size() << endl;
    cout << triangulated.col(1) << endl;
    //cout << triangulated << endl;
    
    
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


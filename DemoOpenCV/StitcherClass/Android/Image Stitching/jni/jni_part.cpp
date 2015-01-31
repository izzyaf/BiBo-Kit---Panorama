#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching/stitcher.hpp>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <list>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

extern "C" {
//JNIEXPORT Mat JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)

//JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindFeatures(JNIEnv*, jobject, jlong im1, jlong im2, jlong im3, jint no_images) {
JNIEXPORT void JNICALL Java_org_opencv_stitchsample_StitchingActivity_stitchimage(
		JNIEnv*, jobject, jint no_images) {
	vector<Mat> imgs;
	bool try_use_gpu = false;
	// New testing
	//Mat& temp1 = *((Mat*) im1);
	//Mat& temp2 = *((Mat*) im2);
	//Mat& pano = *((Mat*) im3);
	Mat pano;
	if (no_images == 0)
		return;
	for (int k = 0; k < no_images; ++k) {
		string id;
		ostringstream convert;
		convert << k;
		id = convert.str();
		Mat img = imread("/storage/emulated/0/DCIM/Camera/im" + id + ".jpg");
		if (img.empty()) break;
		imgs.push_back(img);
	}

	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	vector<int> para;
	para.push_back(CV_IMWRITE_JPEG_QUALITY);
	para.push_back(100);
	imwrite("/storage/emulated/0/DCIM/Camera/result.jpg", pano, para);
	//imwrite("/sdcard/DCIM/Camera/result.jpg", pano, para);
}

}


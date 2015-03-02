/*
 * main.cpp
 *
 *  Created on: Feb 25, 2015
 *      Author: nvkhoi
 */

#include "Stitcher.h"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <opencv2/stitching/stitcher.hpp>

std::string inFileName = "inputpath.txt", outFileName = "runtime.txt";
std::ifstream ifs;
std::fstream ofs;
std::string dst;
int num_images;
std::vector<cv::Mat> vImg(num_images);

bool fileExists(const std::string& filename) {
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
		return true;
	return false;
}

cv::Mat inputMatchMask(const int& size, std::string& pairwise_path) {
	cv::Mat matchMask(size, size, CV_8U, cv::Scalar(0));
	pairwise_path = pairwise_path + "/pairwise.txt";
	if (fileExists(pairwise_path.c_str())) {
		std::ifstream pairwise(pairwise_path.c_str(), std::ifstream::in);
		while (true) {
			int i, j;
			pairwise >> i >> j;
			if (pairwise.eof())
				break;
			matchMask.at<char>(i, j) = 1;
		}
	}
	return matchMask;
}
//Left to Right, Up to Down
std::vector<std::vector<cv::Rect> > createROIS(const int& rows, const int& cols,
		const float& percentage, const cv::Mat& sample) {
	std::vector<std::vector<cv::Rect> > retVal;
	std::vector<cv::Rect> each;
	int perRows = int(sample.rows * percentage), perCols = int(
			sample.cols * percentage);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			cv::Rect tmp(0, sample.cols - perCols, sample.rows, perCols);
			each.push_back(tmp);
			tmp = cv::Rect(0, 0, sample.rows, perCols);
			each.push_back(tmp);
			tmp = cv::Rect(sample.rows - perRows, perCols, perRows,
					sample.cols - 2 * perCols);
			each.push_back(tmp);
			tmp = cv::Rect(0, perCols, perRows, sample.cols - 2 * perCols);
			each.push_back(tmp);
			retVal.push_back(each);
			each.clear();
			/*		if (i == 0) {
			 for (int j = 0; j < cols; j++)
			 if (j == 0) {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, 0, perRows,
			 sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else if (j == cols - 1) {
			 cv::Rect tmp(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, perCols, perRows,
			 sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, perCols, perRows,
			 sample.cols - 2 * perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 }
			 } else if (i == rows - 1) {
			 for (int j = 0; j < cols; j++)
			 if (j == 0) {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, 0, perRows, sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else if (j == cols - 1) {
			 cv::Rect tmp(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, perCols, perRows, sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, perCols, perRows,
			 sample.cols - 2 * perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 }
			 } else {
			 for (int j = 0; j < cols; j++)
			 if (j == 0) {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, 0, perRows,
			 sample.cols - perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, 0, perRows, sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else if (j == cols - 1) {
			 cv::Rect tmp(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, perCols, perRows,
			 sample.cols - perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, perCols, perRows, sample.cols - perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 } else {
			 cv::Rect tmp(0, sample.cols - perCols, sample.rows,
			 perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, 0, sample.rows, perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(sample.rows - perRows, perCols, perRows,
			 sample.cols - 2 * perCols);
			 each.push_back(tmp);
			 tmp = cv::Rect(0, perCols, perRows,
			 sample.cols - 2 * perCols);
			 each.push_back(tmp);
			 retVal.push_back(each);
			 }*/
		}
	}
	return retVal;
}

int main() {
	ifs.open(inFileName.c_str(), std::ifstream::in);
	cv::setBreakOnError(false);
	//std::cout << cv::getNumThreads() << " " << cv::getNumberOfCPUs() << "\n";
	//cv::setNumThreads(16);
	//cv::setUseOptimized(true);
	int num_tests;
	ifs >> num_tests;
	long long start, end;
	ofs.open(outFileName.c_str(), std::fstream::out);
	ofs << std::left << std::setw(10) << "Test case" << std::setw(35)
			<< "1st time" << std::setw(35) << "2st time" << '\n';
	ofs.close();
	for (int i = 0; i < num_tests; i++) {
		ofs.open(outFileName.c_str(), std::fstream::app);
		ifs >> num_images >> dst;
		vImg.resize(num_images);
		ofs << std::left << std::setw(10) << dst;
		std::string pairwise_path = dst;
		cv::Mat matchMask = inputMatchMask(num_images, pairwise_path);
		for (int i = 0; i < num_images; i++) {
			std::string tmp;
			ifs >> tmp;
			vImg[i] = cv::imread(tmp);
		}
		cv::Mat result;
		/*cv::Stitcher stitch = cv::Stitcher::createDefault(false);
		 if (matchMask.at<unsigned char>(0, 1) == 1
		 || matchMask.at<unsigned char>(1, 0) == 1)
		 stitch.setMatchingMask(matchMask);
		 stitch.setFeaturesFinder(new cv::detail::OrbFeaturesFinder(cv::Size(3, 1), 1500, 1.0f,
		 5));
		 stitch.setRegistrationResol(0.3);
		 stitch.setPanoConfidenceThresh(0.6);
		 stitch.setSeamEstimationResol(0.08);
		 stitch.setWarper(new cv::CylindricalWarper());
		 stitch.setSeamFinder(new cv::detail::GraphCutSeamFinder(
		 cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD));
		 stitch.setExposureCompensator(cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN));
		 start = cv::getTickCount();
		 stitch.stitch(vImg, result);
		 end = cv::getTickCount();
		 cv::string tmp = dst+"/result1.jpg";
		 cv::imwrite(tmp, result);
		 ofs << std::left << std::setw(15)
		 << (float(end) - float(start)) / cv::getTickFrequency();*/

		//std::vector<std::vector<cv::Rect> > ROIS = createROIS(3, 12, 0.4, vImg[0]);
		Stitcher stitcher(Stitcher::FAST);
		start = cv::getTickCount();
		Stitcher::ReturnCode status = stitcher.stitch(vImg, matchMask, result);
		end = cv::getTickCount();
		ofs << stitcher.toString();
		ofs << std::left << std::setw(10)
				<< (float(end) - float(start)) / cv::getTickFrequency();
		if (status != Stitcher::OK) {
			cv::Mat retry;
			stitcher = Stitcher(Stitcher::DEFAULT);
			matchMask = cv::Mat(num_images, num_images, CV_8U, cv::Scalar(0));
			start = cv::getTickCount();
			Stitcher::ReturnCode retryStatus = stitcher.stitch(vImg, matchMask, retry);
			end = cv::getTickCount();
			if ((result.rows*result.cols)<(retry.rows*retry.cols)) result=retry;
			ofs << stitcher.toString();
			ofs << std::left << std::setw(10)
					<< (float(end) - float(start)) / cv::getTickFrequency()
					<< '\n';
		} else
			ofs << '\n';
		dst = dst + "/result.jpg";
		cv::imwrite(dst, result);
		vImg.clear();
		stitcher.collectGarbage();

		ofs.close();
		std::cout << std::endl;
	}
	ifs.close();
	return 0;
}


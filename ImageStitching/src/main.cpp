/*
 * main.cpp
 *
 *  Created on: Feb 25, 2015
 *      Author: nvkhoi
 */

#include "Stitcher.h"

std::string uploadDir = "./uploads/", publicDir = "./public/";
std::string workingDir;

int main(int argc, char* argv[]) {
	cv::setBreakOnError(true);

	if (argc < 2)
		return -1;
	workingDir = argv[1];
	Stitcher stitcher;
	std::string dst = publicDir + workingDir;
	stitcher.set_dst(dst);
	workingDir = uploadDir + workingDir + "/";
	stitcher.feed(workingDir);
	long long start = cv::getTickCount();
	stitcher.stitch();
	long long end = cv::getTickCount();
	std::cout << (double(end)-start)/cv::getTickFrequency();

	return 0;
}

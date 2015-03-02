/*
 * Stitcher.h
 *
 *  Created on: Feb 25, 2015
 *      Author: nvkhoi
 */

#ifndef SRC_STITCHER_H_
#define SRC_STITCHER_H_

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

bool sortX(cv::Point, cv::Point);
bool sortY(cv::Point, cv::Point);

class Stitcher {
private:
	double registration_resol, seam_estimation_resol, compositing_resol,
			confidence_threshold;
	int expos_comp_type, blend_type;
	float blend_strength;
	int num_images;
	double seam_work_aspect;
	double work_scale;
	std::vector<cv::Mat> full_img;
	std::vector<cv::Mat> img;
	float warped_image_scale;
	std::vector<cv::Mat> images;
	std::vector<cv::Size> full_img_sizes;
	enum WarpType {
		PLANE,
		CYLINDRICAL,
		SPHERICAL,
		FISHEYE,
		STEREOGRAPHIC,
		COMPRESSEDPLANEA2B1,
		COMPRESSEDPLANEA15B1,
		COMPRESSEDPORTRAITA2B1,
		COMPRESSEDPORTRAITA15B1,
		PANINIA2B1,
		PANINIA15B1,
		PANINIPORTRAITA2B1,
		PANINIPORTRAITA15B1,
		MERCATOR,
		TRANSVERSEMERCATOR
	};
	enum WarpType warp_type;
	enum SeamFindType {
		NO, VORONOI, GC_COLOR, GC_COLORGRAD, DP_COLOR, DP_COLORGRAD
	};
	enum SeamFindType seam_find_type;
	int registration(
			const std::vector<cv::Mat>&, const cv::Mat&, std::vector<cv::detail::CameraParams>&);
	cv::Mat compositing(std::vector<cv::detail::CameraParams>&);
	cv::Mat crop(const cv::Mat&);
	bool checkInteriorExterior(const cv::Mat&, const cv::Rect&, int&, int&,
			int&, int&);
	std::string status;
public:
	enum Mode {
		DEFAULT, FAST, PREVIEW
	};
	enum ReturnCode{
		OK, FAILED, NOT_ENOUGH, NEED_MORE
	};
	Stitcher();
	Stitcher(const int& mode);
	enum ReturnCode stitch(const std::vector<cv::Mat>&, const cv::Mat& , cv::Mat&);
	std::string toString();
	void collectGarbage();
	virtual ~Stitcher();
};

#endif /* SRC_STITCHER_H_ */

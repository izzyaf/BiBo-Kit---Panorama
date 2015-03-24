/*
 * Stitcher.h
 *
 *  Created on: Feb 25, 2015
 *      Author: nvkhoi
 */

#ifndef SRC_STITCHER_H_
#define SRC_STITCHER_H_

#include <bits/stdc++.h>
#include <sys/stat.h>

#include <exiv2/exiv2.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

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
//#include <opencv2/stitching/detail/warpers.hpp>
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
	enum Mode {
		DEFAULT, FAST, PREVIEW
	};
	Stitcher(const int& mode);
	void set_matching_mask(const std::string& file_name,
			std::vector<std::pair<int, int> >& edge_list);
	int rotate_img(const std::string&);
	int registration(std::vector<cv::detail::CameraParams>&);
	cv::Mat compositing(std::vector<cv::detail::CameraParams>&);
	enum ReturnCode {
		OK, FAILED, NOT_ENOUGH, NEED_MORE
	};
	enum ReturnCode stitching_process(cv::Mat&);
	cv::Mat crop(const cv::Mat&) __attribute__ ((deprecated));bool checkInteriorExterior(
			const cv::Mat&, const cv::Rect&, int&, int&, int&, int&)
					__attribute__ ((deprecated));

	std::string status;
	std::string result_dst;
	cv::Mat matching_mask;

public:
	Stitcher();
	void set_dst(const std::string&);
	std::string get_dst();
	void feed(const std::string&);
	void stitch();
	std::string to_string();
	virtual ~Stitcher();
};

#endif /* SRC_STITCHER_H_ */

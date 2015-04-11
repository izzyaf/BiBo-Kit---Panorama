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
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

#define ON_LOGGER true

class Stitcher {

private:
	/*
	 * Registration resolution: parameter for resizing images to find features
	 * Seam estimation resolution*: parameter for seam estimation
	 * Compositing resolution: affect output resolution
	 * Confidence threshold: used for pairwise matching
	 */
	double registration_resol, seam_estimation_resol, compositing_resol,
			confidence_threshold;
	/*
	 * expos_comp_type: enum contains type of Exposure Compensator
	 * blend_type: type of blender
	 */
	int expos_comp_type, blend_type;
	int num_images; //number of input images
	double seam_work_aspect; //for warping images
	double work_scale; //finding features and blending
	float warped_image_scale; //blending
	std::vector<cv::Mat> full_img; //original images
	std::vector<cv::Mat> img; //temporary images used for finding features and blending
	std::vector<cv::Mat> images; //temporary images used for warping
	std::vector<cv::Size> full_img_sizes; //sizes of original images
	enum ReturnCode {
		OK, NOT_ENOUGH, FAILED, NEED_MORE
	};
	std::pair<ReturnCode, double> status; // status of stitching process: failed, success or not enough
	std::string result_dst; ////determine the input and output directory
	cv::Mat matching_mask; //contain pairs of image that can be stitched together
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

	//Stitcher class's initialization with argument
	void init();
	//Input matching mask from file
	void set_matching_mask(const std::string&,
			std::vector<std::pair<int, int> >&);

	//Rotate images for better stitching
	int rotate_img(const std::string&);

	//Find image's features for matching
	void find_features(std::vector<cv::detail::ImageFeatures>&);

	//Match pairs of images based on features
	void match_pairwise(std::vector<cv::detail::ImageFeatures>&,
			std::vector<cv::detail::MatchesInfo>&);

	//Extract biggest component from pairwise matching
	void extract_biggest_component(std::vector<cv::detail::ImageFeatures>&,
			std::vector<cv::detail::MatchesInfo>&);

	//Estimate camera
	void estimate_camera(std::vector<cv::detail::ImageFeatures>&,
			std::vector<cv::detail::MatchesInfo>&,
			std::vector<cv::detail::CameraParams>&);

	//Refine
	void refine_camera(const std::vector<cv::detail::ImageFeatures>&,
			const std::vector<cv::detail::MatchesInfo>&,
			std::vector<cv::detail::CameraParams>&);

	//First stage of stitching, do needed calculation for stitching
	int registration(std::vector<cv::detail::CameraParams>&);

	//Create warper that effect the "style" of output
	void create_warper(std::vector<cv::Ptr<cv::detail::RotationWarper> >&,
			cv::Ptr<cv::WarperCreator>&);

	//Warp all remain images from pairwise matching
	std::vector<cv::Mat> warp_img(std::vector<cv::Point>&,
			std::vector<cv::Ptr<cv::detail::RotationWarper> >&,
			std::vector<cv::Size>&, std::vector<cv::Mat>&,
			std::vector<cv::detail::CameraParams>&,
			cv::Ptr<cv::detail::ExposureCompensator>&);

	//Find seam for stitching and blending
	void find_seam(std::vector<cv::Mat>&, const std::vector<cv::Point>&,
			std::vector<cv::Mat>&);

	//Resize mask for blending
	double resize_mask(std::vector<cv::Ptr<cv::detail::RotationWarper> >&,
			cv::Ptr<cv::WarperCreator>&, std::vector<cv::Point>&,
			std::vector<cv::Size>&, std::vector<cv::detail::CameraParams>&);

	//Prepare blend
	cv::Ptr<cv::detail::Blender> prepare_blender(const std::vector<cv::Point>&,
			const std::vector<cv::Size>&);

	//Stitch and blend output pano
	void blend_img(double&, std::vector<cv::Ptr<cv::detail::RotationWarper> >&,
			cv::Ptr<cv::detail::ExposureCompensator>&, std::vector<cv::Point>&,
			std::vector<cv::Mat>&, cv::Ptr<cv::detail::Blender>&,
			std::vector<cv::detail::CameraParams>&, cv::Mat&);

	//Final stage of stitching, do all work basing on first stage output
	cv::Mat compositing(std::vector<cv::detail::CameraParams>&);

	//The whole process combing first and second stage
	void stitching_process(cv::Mat&);

	void collect_garbage();

public:

	//Stitcher class's constructor with no argument
	Stitcher();

	//Set input and output directories, both have same name
	void set_dst(const std::string&);
	//Get output directory's name
	std::string get_dst();
	//Input images and do some pre-calculation
	void feed(const std::string&);

	//Stitch images into one panorama including retry
	void stitch();

	std::string get_status();

	//Stitcher class's destructor
	virtual ~Stitcher();
};

#endif /* SRC_STITCHER_H_ */

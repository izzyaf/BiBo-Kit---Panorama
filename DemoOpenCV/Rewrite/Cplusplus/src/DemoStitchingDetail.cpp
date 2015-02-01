//============================================================================
// Name        : DemoStitchingDetail.cpp
// Author      : nvkhoi
// Version     :
// Copyright   : Your copyright notice
// Description : Stitching Details, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <string>
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

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default constant parameters
vector<string> imgNames;
int num_images;
bool preview = false, use_gpu = false, correct_wave = true, save_graph = false;
double registration_resol = 0.6, seam_estimation_resol = 0.1,
		compositing_resol = -1.0, confidence_threshold = 1.0;
double seam_work_aspect = 1, work_scale = 1;
float warped_image_scale;
string features_finder_type = "surf", bundleadjuster_cost_func = "ray",
		bundleadjuster_refine_mask = "xxxxx", warp_type = "spherical",
		seam_find_type = "gc_color";
WaveCorrectKind wave_correct_type = detail::WAVE_CORRECT_HORIZ;
string graph_dest = "graph.txt", result_dest = "result.jpg";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS, blend_type =
		Blender::MULTI_BAND;
float match_conf = 0.3f, blend_strength = 5;

static int parseInputArg(int argc, char** argv) {
	if (argc == 1)
		return -1;
	for (int i = 1; i < argc; ++i)
		imgNames.push_back(argv[i]);
	num_images = static_cast<int>(imgNames.size());
	return 0;
}

//Finding features
Ptr<FeaturesFinder> initFeaturesFinder() {
	Ptr<FeaturesFinder> finder;
	if (features_finder_type == "surf") {
		if (use_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			finder = new SurfFeaturesFinderGpu();
		else
			finder = new SurfFeaturesFinder();
	} else if (features_finder_type == "orb") {
		finder = new OrbFeaturesFinder();
	} else {
		cout << "Unknown 2D features type: '" << features_finder_type << "'.\n";
		return NULL;
	}
	return finder;
}
vector<ImageFeatures> findFeatures(vector<Size>& full_img_sizes,
		vector<Mat>& images) {
	vector<ImageFeatures> features(num_images);
	Mat full_img, img;
	double seam_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false;
	Ptr<FeaturesFinder> finder = initFeaturesFinder();

	//Reading images, finding features of each one and resizing if necessary
	for (int i = 0; i < num_images; ++i) {
		full_img = imread(imgNames[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty()) {
			//return NULL;
		}

		if (registration_resol < 0) {
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		} else {
			if (!is_work_scale_set) {
				work_scale = min(1.0,
						sqrt(
								registration_resol * 1e6
										/ full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set) {
			seam_scale = min(1.0,
					sqrt(seam_estimation_resol * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;

		resize(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();
	return features;
}

//Pairwise matching
vector<MatchesInfo> pairwiseMatching(vector<ImageFeatures>& features,
		vector<Mat>& images, vector<Size>& full_img_sizes) {
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(use_gpu, match_conf);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();

	// Check if we should save matches graph
	if (save_graph) {
		ofstream f(graph_dest.c_str());
		f
				<< matchesGraphAsString(imgNames, pairwise_matches,
						confidence_threshold);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches,
			confidence_threshold);
	vector<Mat> img_subset;
	vector<string> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i) {
		img_names_subset.push_back(imgNames[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	imgNames = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(imgNames.size());
	if (num_images < 2) {
		//return NULL;
	}
	return pairwise_matches;
}

//Estimate and refine camera parameters
Ptr<detail::BundleAdjusterBase> initBundleAdjuster() {
	Ptr<detail::BundleAdjusterBase> adjuster;
	if (bundleadjuster_cost_func == "reproj")
		adjuster = new detail::BundleAdjusterReproj();
	else if (bundleadjuster_cost_func == "ray")
		adjuster = new detail::BundleAdjusterRay();
	else {
		cout << "Unknown bundle adjustment cost function: '"
				<< bundleadjuster_cost_func << "'.\n";
		return NULL;
	}

	return adjuster;
}
Mat_<uchar> refineMask() {
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (bundleadjuster_refine_mask[0] == 'x')
		refine_mask(0, 0) = 1;

	if (bundleadjuster_refine_mask[1] == 'x')
		refine_mask(0, 1) = 1;

	if (bundleadjuster_refine_mask[2] == 'x')
		refine_mask(0, 2) = 1;

	if (bundleadjuster_refine_mask[3] == 'x')
		refine_mask(1, 1) = 1;

	if (bundleadjuster_refine_mask[4] == 'x')
		refine_mask(1, 2) = 1;

	return refine_mask;
}
Ptr<detail::BundleAdjusterBase> createBundleAdjuster() {
	Ptr<detail::BundleAdjusterBase> adjuster = initBundleAdjuster();
	adjuster->setConfThresh(confidence_threshold);
	Mat_<uchar> refine_mask = refineMask();
	adjuster->setRefinementMask(refine_mask);
	return adjuster;
}
vector<CameraParams> estimate_refineCamera(
		const vector<MatchesInfo>& pairwise_matches,
		vector<ImageFeatures>& features) {
	vector<CameraParams> cameras;
	HomographyBasedEstimator estimator;
	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i) {
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	Ptr<detail::BundleAdjusterBase> adjuster = createBundleAdjuster();
	(*adjuster)(features, pairwise_matches, cameras);

	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i) {
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1]
				+ focals[focals.size() / 2]) * 0.5f;

	if (correct_wave) {
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R);
		waveCorrect(rmats, wave_correct_type);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}
	return cameras;
}

//Warping images (auxiliary)...
Ptr<WarperCreator> initWarperCreator() {
	Ptr<WarperCreator> warper_creator;
	if (use_gpu && gpu::getCudaEnabledDeviceCount() > 0) {
		if (warp_type == "plane")
			warper_creator = new cv::PlaneWarperGpu();
		else if (warp_type == "cylindrical")
			warper_creator = new cv::CylindricalWarperGpu();
		else if (warp_type == "spherical")
			warper_creator = new cv::SphericalWarperGpu();
	} else {
		if (warp_type == "plane")
			warper_creator = new cv::PlaneWarper();
		else if (warp_type == "cylindrical")
			warper_creator = new cv::CylindricalWarper();
		else if (warp_type == "spherical")
			warper_creator = new cv::SphericalWarper();
		else if (warp_type == "fisheye")
			warper_creator = new cv::FisheyeWarper();
		else if (warp_type == "stereographic")
			warper_creator = new cv::StereographicWarper();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = new cv::CompressedRectilinearWarper(2, 1);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5,
					1);
		else if (warp_type == "paniniA2B1")
			warper_creator = new cv::PaniniWarper(2, 1);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = new cv::PaniniWarper(1.5, 1);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = new cv::PaniniPortraitWarper(2, 1);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
		else if (warp_type == "mercator")
			warper_creator = new cv::MercatorWarper();
		else if (warp_type == "transverseMercator")
			warper_creator = new cv::TransverseMercatorWarper();
	}
	if (warper_creator.empty()) {
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return NULL;
	}
	return warper_creator;
}
vector<Mat> warpImages(vector<Mat>& images, const vector<CameraParams>& cameras,
		vector<Point>& corners, vector<Size>& sizes,
		Ptr<RotationWarper>& warper, vector<Mat>& masks_warped) {

	vector<Mat> images_warped(num_images);
	vector<Mat> masks(num_images);
	// Prepare images masks
	for (int i = 0; i < num_images; ++i) {
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}
	// Warp images and their masks
	for (int i = 0; i < num_images; ++i) {
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float) seam_work_aspect;
		K(0, 0) *= swa;
		K(0, 2) *= swa;
		K(1, 1) *= swa;
		K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR,
				BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT,
				masks_warped[i]);
	}

	// Release unused memory
	images.clear();
	//images_warped.clear();
	masks.clear();
	return images_warped;
}

//Estimating seams
Ptr<SeamFinder> initSeamFinder() {
	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type == "gc_color") {
		if (use_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(
					GraphCutSeamFinderBase::COST_COLOR);
		else
			seam_finder = new detail::GraphCutSeamFinder(
					GraphCutSeamFinderBase::COST_COLOR);
	} else if (seam_find_type == "gc_colorgrad") {
		if (use_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(
					GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
			seam_finder = new detail::GraphCutSeamFinder(
					GraphCutSeamFinderBase::COST_COLOR_GRAD);
	} else if (seam_find_type == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);

	if (seam_finder.empty()) {
		cout << "Can't create the following seam finder '" << seam_find_type
				<< "'\n";
		return NULL;
	}
	return seam_finder;
}
void estimateSeam(const vector<Mat>& images_warped, vector<Point>& corners,
		vector<Mat>& masks_warped) {
	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	Ptr<SeamFinder> seam_finder = initSeamFinder();
	seam_finder->find(images_warped_f, corners, masks_warped);
	images_warped_f.clear();
}

//Composing and blending
Ptr<ExposureCompensator> createCompensator(const vector<Mat>& images_warped,
		vector<Point>& corners, vector<Mat>& masks_warped) {
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(
			expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);
	return compensator;
}
bool correctComposeScaleSet(const Mat& full_img, double& compose_work_aspect,
		const Ptr<WarperCreator>& warper_creator,
		const vector<Size>& full_img_sizes, double& compose_scale,
		Ptr<RotationWarper>& warper, vector<CameraParams>& cameras,
		vector<Point>& corners, vector<Size>& sizes) {
	if (compositing_resol > 0)
		compose_scale = min(1.0,
				sqrt(compositing_resol * 1e6 / full_img.size().area()));
	// Compute relative scales
	compose_work_aspect = compose_scale / work_scale;
	// Update warped image scale
	warped_image_scale *= static_cast<float>(compose_work_aspect);
	warper = warper_creator->create(warped_image_scale);
	// Update corners and sizes
	for (int i = 0; i < num_images; ++i) {
		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx *= compose_work_aspect;
		cameras[i].ppy *= compose_work_aspect;

		// Update corner and size
		Size sz = full_img_sizes[i];
		if (std::abs(compose_scale - 1) > 1e-1) {
			sz.width = cvRound(full_img_sizes[i].width * compose_scale);
			sz.height = cvRound(full_img_sizes[i].height * compose_scale);
		}

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(sz, K, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}
	return true;
}
Ptr<Blender> initBlender(const float& blend_width) {
	Ptr<Blender> blender = Blender::createDefault(blend_type, use_gpu);
	if (blend_width < 1.f)
		blender = Blender::createDefault(Blender::NO, use_gpu);
	else if (blend_type == Blender::MULTI_BAND) {
		MultiBandBlender* mb =
				dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
		mb->setNumBands(
				static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	} else if (blend_type == Blender::FEATHER) {
		FeatherBlender* fb =
				dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
		fb->setSharpness(1.f / blend_width);
	}

	return blender;
}
Ptr<Blender> correctBlender(vector<Size>& sizes, vector<Point>& corners) {
	Size dst_sz = resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength
			/ 100.f;
	Ptr<Blender> blender = initBlender(blend_width);
	blender->prepare(corners, sizes);
	return blender;
}
Mat composePano(Ptr<RotationWarper> warper,
		const Ptr<WarperCreator>& warper_creator, vector<CameraParams>& cameras,
		const vector<Size>& full_img_sizes, vector<Point>& corners,
		vector<Size>& sizes, Ptr<ExposureCompensator>& compensator,
		const vector<Mat>& masks_warped) {
	Mat full_img, img;
	Ptr<Blender> blender;
	bool is_compose_scale_set = false;
	double compose_scale = 1;
	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	double compose_work_aspect = 1;
	for (int img_idx = 0; img_idx < num_images; ++img_idx) {

		// Read image and resize it if necessary
		full_img = imread(imgNames[img_idx]);
		if (!is_compose_scale_set) {
			is_compose_scale_set = correctComposeScaleSet(full_img,
					compose_work_aspect, warper_creator, full_img_sizes,
					compose_scale, warper, cameras, corners, sizes);
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img, img, Size(), compose_scale, compose_scale);
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT,
				img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST,
				BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;

		if (blender.empty()) {
			blender = correctBlender(sizes, corners);
		}
		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	}
	Mat result_mask, result;
	blender->blend(result, result_mask);
	return result;
}

//Panorama Stiching Process
vector<CameraParams> registrationStage(vector<Size>& full_img_sizes,
		vector<Mat>& images) {
	vector<ImageFeatures> features = findFeatures(full_img_sizes, images);
	vector<MatchesInfo> pairwise_matches = pairwiseMatching(features, images,
			full_img_sizes);
	vector<CameraParams> cameras = estimate_refineCamera(pairwise_matches,
			features);
	return cameras;
}
Mat compositingStage(vector<Mat> images, vector<CameraParams> cameras,
		const vector<Size>& full_img_sizes) {
	Ptr<WarperCreator> warper_creator = initWarperCreator();
	Ptr<RotationWarper> warper = warper_creator->create(
			static_cast<float>(warped_image_scale * seam_work_aspect));
	vector<Point> corners(num_images);
	vector<Mat> masks_warped(num_images);
	vector<Size> sizes(num_images);
	vector<Mat> images_warped = warpImages(images, cameras, corners, sizes,
			warper, masks_warped);
	Ptr<ExposureCompensator> compensator = createCompensator(images_warped,
			corners, masks_warped);
	estimateSeam(images_warped, corners, masks_warped);
	Mat result = composePano(warper, warper_creator, cameras, full_img_sizes,
			corners, sizes, compensator, masks_warped);
	return result;
}

int main(int argc, char* argv[]) {
	long long start = getTickCount();
	cv::setBreakOnError(true);

	int retval = parseInputArg(argc, argv);
	if (retval)
		return retval;
	// Check if have enough images
	if (num_images < 2)
		return -1;

	//Registration
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	vector<CameraParams> cameras = registrationStage(full_img_sizes, images);

	//Compositing
	Mat result = compositingStage(images, cameras, full_img_sizes);
	imwrite(result_dest, result);
	long long end = getTickCount();
	printf("%.6lf\n", (float(end) - float(start)) / getTickFrequency());

	return 0;
}

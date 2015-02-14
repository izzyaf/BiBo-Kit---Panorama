//============================================================================
// Name        : Stitcher.cpp
// Author      : nvkhoi
// Version     :
// Copyright   : Your copyright notice
// Description : Stitching Details, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
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
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace cv::detail;

string inFileName = "inputpath.txt", outFileName = "runtimeROI.txt";
ifstream ifs;
fstream ofs;
string dst;
int num_images;
vector<string> imgNames;

class Stitcher {
private:
	bool useGPU, do_wave_correct;
	double registration_resol, seam_estimation_resol, compositing_resol,
			confidence_threshold;
	string features_finder_type, bundleadjuster_cost_func,
			bundleadjuster_refine_mask, warp_type, seam_find_type;
	WaveCorrectKind wave_correct;
	string result_dest;
	int expos_comp_type, blend_type;
	float match_conf, blend_strength;
	int num_images;
	double seam_work_aspect;
	double work_scale;
	vector<Mat> full_img;
	vector<Mat> img;
	float warped_image_scale;
	vector<Mat> images;
	vector<Size> full_img_sizes;
public:
	Stitcher(const bool& isFast, const bool& isPreview) {
		warped_image_scale = 1.0;
		result_dest = "result.jpg";
		num_images = 0;
		blend_type = Blender::MULTI_BAND;
		match_conf = 0.3f;
		blend_strength = 5;
		bundleadjuster_cost_func = "ray";
		bundleadjuster_refine_mask = "xxxxx";
		features_finder_type = "surf";
		seam_work_aspect = 1;
		work_scale = 1;
		useGPU = false;
		do_wave_correct = true;
		wave_correct = detail::WAVE_CORRECT_HORIZ;
		if (isFast) {
			registration_resol = 0.3;
			seam_estimation_resol = 0.08;
			confidence_threshold = 0.6;
			warp_type = "cylindrical";
			seam_find_type = "dp_colorgrad";
			expos_comp_type = ExposureCompensator::GAIN;
		} else {
			registration_resol = 0.6;
			seam_estimation_resol = 0.1;
			confidence_threshold = 1.0;
			warp_type = "spherical";
			seam_find_type = "gc_color";
			expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
		}
		if (isPreview)
			compositing_resol = 0.6;
		else
			compositing_resol = -1.0;

	}

//Finding features
	void findFeatures(const vector<string> imgNames,
			vector<ImageFeatures>& features) {
		//Mat full_img, img;
		double seam_scale = 1;
		bool is_work_scale_set = false, is_seam_scale_set = false;
		Ptr<FeaturesFinder> finder;
		finder = new SurfFeaturesFinder();
		if (features_finder_type == "surf") {
			if (useGPU && gpu::getCudaEnabledDeviceCount() > 0)
				finder = new SurfFeaturesFinderGpu();
			else
				finder = new SurfFeaturesFinder();
		} else if (features_finder_type == "orb") {
			finder = new OrbFeaturesFinder();
		} else {
			return;
		}

#pragma omp parallel for
		for (int i = 0; i < num_images; ++i) {
			full_img[i] = imread(imgNames[i]);
			full_img_sizes[i] = full_img[i].size();

			/*		if (full_img[i].empty())
			 return;*/
			if (registration_resol < 0) {
				img[i] = full_img[i];
				work_scale = 1;
				is_work_scale_set = true;
			} else {
				if (!is_work_scale_set) {
					work_scale = min(1.0,
							sqrt(
									registration_resol * 1e6
											/ full_img[i].size().area()));
					is_work_scale_set = true;
				}
				resize(full_img[i], img[i], Size(), work_scale, work_scale);
			}
			if (!is_seam_scale_set) {
				seam_scale = min(1.0,
						sqrt(
								seam_estimation_resol * 1e6
										/ full_img[i].size().area()));
				seam_work_aspect = seam_scale / work_scale;
				is_seam_scale_set = true;
			}

			(*finder)(img[i], features[i]);
			features[i].img_idx = i;

			resize(full_img[i], img[i], Size(), seam_scale, seam_scale);
			images[i] = img[i].clone();
		}

		finder->collectGarbage();
		//full_img.release();
		//img.release();
	}

//Pairwise matching
	int pairwiseMatching(vector<ImageFeatures>& features,
			vector<CameraParams>& cameras,
			vector<MatchesInfo>& pairwise_matches, const Mat& matchMask) {

		BestOf2NearestMatcher matcher(useGPU, match_conf);

		matcher(features, pairwise_matches, matchMask);

		//matcher(features, pairwise_matches);
		matcher.collectGarbage();

		// Check if we should save matches graph
		/*if (save_graph) {
		 ofstream f(graph_dest.c_str());
		 f
		 << matchesGraphAsString(imgNames, pairwise_matches,
		 confidence_threshold);
		 }
		 */
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
		int tmp = static_cast<int>(imgNames.size());
		if (tmp < 2) {
			return -1;
		} else if (tmp < num_images) {
			ofs << left << setw(3) << tmp << "/" << setw(3) << num_images;
			num_images = tmp;
			return 0;
		} else
			return 1;

	}

//Estimate and refine camera parameters
	void estimate_refineCamera(vector<MatchesInfo> pairwise_matches,
			vector<ImageFeatures>& features, vector<CameraParams>& cameras) {
		HomographyBasedEstimator estimator;
		estimator(features, pairwise_matches, cameras);
		for (size_t i = 0; i < cameras.size(); ++i) {
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
		}
		Ptr<detail::BundleAdjusterBase> adjuster =
				new detail::BundleAdjusterRay();
		if (bundleadjuster_cost_func == "reproj")
			adjuster = new detail::BundleAdjusterReproj();
		else if (bundleadjuster_cost_func == "ray")
			adjuster = new detail::BundleAdjusterRay();

		/*	else {
		 cout << "Unknown bundle adjustment cost function: '"
		 << bundleadjuster_cost_func << "'.\n";
		 return;
		 }*/
		adjuster->setConfThresh(confidence_threshold);
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

		adjuster->setRefinementMask(refine_mask);
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
			warped_image_scale =
					static_cast<float>(focals[focals.size() / 2 - 1]
							+ focals[focals.size() / 2]) * 0.5f;

		if (do_wave_correct) {
			vector<Mat> rmats;
			for (size_t i = 0; i < cameras.size(); ++i)
				rmats.push_back(cameras[i].R);
			waveCorrect(rmats, wave_correct);
			for (size_t i = 0; i < cameras.size(); ++i)
				cameras[i].R = rmats[i];
		}
	}

//Warping images (auxiliary)...
	void createWarper(Ptr<WarperCreator>& warper_creator,
			vector<Ptr<RotationWarper> >& warper) {
		// Warp images and their masks
		if (useGPU && gpu::getCudaEnabledDeviceCount() > 0) {
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
				warper_creator = new cv::CompressedRectilinearPortraitWarper(2,
						1);
			else if (warp_type == "compressedPlanePortraitA1.5B1")
				warper_creator = new cv::CompressedRectilinearPortraitWarper(
						1.5, 1);
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
			return;
		}
#pragma omp parallel for
		for (int i = 0; i < num_images; i++)
			warper[i] = warper_creator->create(
					static_cast<float>(warped_image_scale * seam_work_aspect));
	}

	void warpImages(const vector<CameraParams>& cameras, vector<Point>& corners,
			vector<Size>& sizes, vector<Ptr<RotationWarper> >& warper,
			vector<Mat>& masks_warped, Ptr<ExposureCompensator>& compensator) {

		vector<Mat> images_warped(num_images);
		vector<Mat> masks(num_images);
		// Prepare images masks
#pragma omp parallel for
		for (int i = 0; i < num_images; ++i) {
			masks[i].create(images[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}
		// Warp images and their masks
		vector<Mat> images_warped_f(num_images);
#pragma omp parallel for
		for (int i = 0; i < num_images; ++i) {
			Mat_<float> K;
			cameras[i].K().convertTo(K, CV_32F);
			float swa = (float) seam_work_aspect;
			K(0, 0) *= swa;
			K(0, 2) *= swa;
			K(1, 1) *= swa;
			K(1, 2) *= swa;

			corners[i] = warper[i]->warp(images[i], K, cameras[i].R,
					INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			sizes[i] = images_warped[i].size();

			warper[i]->warp(masks[i], K, cameras[i].R, INTER_NEAREST,
					BORDER_CONSTANT, masks_warped[i]);
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}
		compensator = ExposureCompensator::createDefault(expos_comp_type);
		compensator->feed(corners, images_warped, masks_warped);
		Ptr<SeamFinder> seam_finder;
		if (seam_find_type == "no")
			seam_finder = new detail::NoSeamFinder();
		else if (seam_find_type == "voronoi")
			seam_finder = new detail::VoronoiSeamFinder();
		else if (seam_find_type == "gc_color") {
			if (useGPU && gpu::getCudaEnabledDeviceCount() > 0)
				seam_finder = new detail::GraphCutSeamFinderGpu(
						GraphCutSeamFinderBase::COST_COLOR);
			else
				seam_finder = new detail::GraphCutSeamFinder(
						GraphCutSeamFinderBase::COST_COLOR);
		} else if (seam_find_type == "gc_colorgrad") {
			if (useGPU && gpu::getCudaEnabledDeviceCount() > 0)
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
			return;
		}
		seam_finder->find(images_warped_f, corners, masks_warped);
		// Release unused memory
		images.clear();
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();
	}

//Composing and blending
	void composePano(vector<Ptr<RotationWarper> >& warper,
			const Ptr<WarperCreator>& warper_creator,
			vector<CameraParams>& cameras, vector<Point>& corners,
			vector<Size>& sizes, Ptr<ExposureCompensator>& compensator,
			const vector<Mat>& masks_warped, Mat& result) {
		//Mat full_img, img;
		Ptr<Blender> blender;
		bool is_compose_scale_set = false;
		double compose_scale = 1;
		double compose_work_aspect = 1;
		if (!is_compose_scale_set) {
			if (compositing_resol > 0)
				compose_scale = min(1.0,
						sqrt(
								compositing_resol * 1e6
										/ full_img[0].size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
#pragma omp parallel for
			for (int i = 0; i < num_images; i++)
				warper[i] = warper_creator->create(warped_image_scale);

			// Update corners and sizes
#pragma omp parallel for
			for (int i = 0; i < num_images; ++i) {
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1) {
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(
							full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper[i]->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (blender.empty()) {
			blender = Blender::createDefault(blend_type, useGPU);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area()))
					* blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, useGPU);
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
			blender->prepare(corners, sizes);
		}
		vector<Mat> mask_warped(num_images), img_warped_s(num_images);
		vector<Mat> dilated_mask(num_images), seam_mask(num_images);
		vector<Mat> mask(num_images), img_warped(num_images);
#pragma omp parallel for
		for (int img_idx = 0; img_idx < num_images; ++img_idx) {
			// Read image and resize it if necessary
			//full_img = imread(imgNames[img_idx]);
			//Mat mask_warped, img_warped_s;
			if (abs(compose_scale - 1) > 1e-1)
				resize(full_img[img_idx], img[img_idx], Size(), compose_scale,
						compose_scale);
			else
				img[img_idx] = full_img[img_idx];
			//full_img[img_idx].release();
			Size img_size = img[img_idx].size();

			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			// Warp the current image
			warper[img_idx]->warp(img[img_idx], K, cameras[img_idx].R,
					INTER_LINEAR, BORDER_REFLECT, img_warped[img_idx]);

			// Warp the current image mask
			mask[img_idx].create(img_size, CV_8U);
			mask[img_idx].setTo(Scalar::all(255));
			warper[img_idx]->warp(mask[img_idx], K, cameras[img_idx].R,
					INTER_NEAREST, BORDER_CONSTANT, mask_warped[img_idx]);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped[img_idx],
					mask_warped[img_idx]);

			img_warped[img_idx].convertTo(img_warped_s[img_idx], CV_16S);
			//img_warped.release();
			//img[img_idx].release();
			//mask.release();

			dilate(masks_warped[img_idx], dilated_mask[img_idx], Mat());
			resize(dilated_mask[img_idx], seam_mask[img_idx],
					mask_warped[img_idx].size());
			mask_warped[img_idx] = seam_mask[img_idx] & mask_warped[img_idx];

			// Blend the current image
			blender->feed(img_warped_s[img_idx], mask_warped[img_idx],
					corners[img_idx]);
		}

		full_img.clear();
		img.clear();
		Mat result_mask;
		blender->blend(result, result_mask);
	}

	vector<CameraParams> registration(const vector<string>& filesNameList,
			const Mat& matchMask) {
		num_images = filesNameList.size();
		full_img.resize(num_images);
		img.resize(num_images);
		images.resize(num_images);
		full_img_sizes.resize(num_images);
		vector<ImageFeatures> features(num_images);
		findFeatures(filesNameList, features);
		vector<CameraParams> cameras;
		vector<MatchesInfo> pairwise_matches;
		pairwiseMatching(features, cameras, pairwise_matches, matchMask);
		estimate_refineCamera(pairwise_matches, features, cameras);
		return cameras;
	}

	Mat compositing(vector<CameraParams>& cameras) {
		Ptr<WarperCreator> warper_creator;
		vector<Ptr<RotationWarper> > warper(num_images);
		createWarper(warper_creator, warper);
		vector<Point> corners(num_images);
		vector<Mat> masks_warped(num_images);
		Ptr<ExposureCompensator> compensator;
		vector<Size> sizes(num_images);
		warpImages(cameras, corners, sizes, warper, masks_warped, compensator);
		Mat result;
		composePano(warper, warper_creator, cameras, corners, sizes,
				compensator, masks_warped, result);
		return result;
	}
	void stitch(const vector<string>& filesNameList, const Mat& matchMask,
			string& dst) {
		vector<CameraParams> cameras = registration(filesNameList, matchMask);
		Mat result = compositing(cameras);
		dst = dst + "/" + result_dest;
		imwrite(dst, result);
	}

};

void inputFilesName() {
	for (int i = 0; i < num_images; i++) {
		string tmp;
		ifs >> tmp;
		imgNames.push_back(tmp);
	}
}

bool fileExists(const string& filename) {
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
		return true;
	return false;
}

Mat inputMatchMask(const int& size, string& pairwise_path) {
	Mat matchMask(size, size, CV_8U, Scalar(0));
	pairwise_path = pairwise_path + "/pairwise.txt";
	if (fileExists(pairwise_path.c_str())) {
		ifstream pairwise(pairwise_path.c_str(), ifstream::in);
		while (true) {
			int i, j;
			pairwise >> i >> j;
			if (pairwise.eof())
				break;
			matchMask.at<char>(i, j) = 1;
		}
	} else {
		for (int i = 0; i < num_images - 1; i++)
			matchMask.at<char>(i, i + 1) = 1;
	}
	return matchMask;
}

int main(int argc, char* argv[]) {
	ifs.open(inFileName.c_str(), ifstream::in);
	setBreakOnError(true);
	int num_tests;
	ifs >> num_tests;
	long long start, end;
	ofs.open(outFileName.c_str(), fstream::out);
	ofs << left << setw(15) << "Test case" << setw(15) << "Running Time" << setw(15)
			<< "Size" << '\n';
	ofs.close();
	Stitcher stitcher(true, false);
	while (num_tests--) {
		ofs.open(outFileName.c_str(), fstream::app);
		ifs >> num_images >> dst;
		ofs << left << setw(15) << dst << setw(15) << num_images;
		inputFilesName();
		start = getTickCount();
		string pairwise_path = dst;
		Mat matchMask = inputMatchMask(num_images, pairwise_path);
		stitcher.stitch(imgNames, matchMask, dst);
		imgNames.clear();
		end = getTickCount();
		ofs << left << setw(15)
				<< (float(end) - float(start)) / getTickFrequency() << '\n';

		ofs.close();
	}
	ifs.close();
	return 0;
}

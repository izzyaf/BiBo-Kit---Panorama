/*
 * Stitcher.cpp
 *
 *  Created on: Feb 25, 2015
 *      Author: nvkhoi
 */

#include "Stitcher.h"

int compareCvSize(const cv::Size& size_1, const cv::Size& size_2) {
	return (size_1.area() < size_2.area());
}

void Stitcher::find_features(std::vector<cv::detail::ImageFeatures>& features) {
#if ON_LOGGER
	printf("Find features with expected: ");
#endif
	work_scale = std::min(1.0,
			sqrt(registration_resol * 1e6 / full_img[0].size().area()));
	double seam_scale = std::min(1.0,
			sqrt(seam_estimation_resol * 1e6 / full_img[0].size().area()));
	seam_work_aspect = seam_scale / work_scale;
	int num_features = int(
			(work_scale * work_scale * full_img[0].size().area()) / 100);
#if ON_LOGGER
	printf("%d\n", num_features);
#endif
	cv::Ptr<cv::detail::FeaturesFinder> finder =
			new cv::detail::OrbFeaturesFinder(cv::Size(3, 1), num_features,
					1.3f, 5);

#pragma omp parallel for
	for (int i = 0; i < num_images; ++i) {
		if (registration_resol <= 0) {
			img[i] = full_img[i];
		} else {
			cv::resize(full_img[i], img[i], cv::Size(), work_scale, work_scale);
		}
		(*finder)(img[i], features[i]);
#if ON_DETAIL
		printf("	i%d %dx%d: %d features\n", i, img[i].rows, img[i].cols,
				int(features[i].keypoints.size()));
#endif
		features[i].img_idx = i;
		cv::resize(full_img[i], img[i], cv::Size(), seam_scale, seam_scale);
		images[i] = img[i].clone();
	}
	img.clear();
	finder->collectGarbage();
}

void Stitcher::extract_biggest_component(
		std::vector<cv::detail::ImageFeatures>& features,
		std::vector<cv::detail::MatchesInfo>& pairwise_matches) {
	// Leave only images we are sure are from the same panorama
	std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches,
			confidence_threshold);
	std::vector<cv::Mat> img_subset(indices.size());
	std::vector<cv::Size> full_img_sizes_subset(indices.size());
	std::vector<cv::Mat> full_img_subset(indices.size());
#if ON_LOGGER
	printf("Biggest component: ");
#endif
#pragma omp parallel for
	for (size_t i = 0; i < indices.size(); ++i) {
#if ON_LOGGER
		printf("%d ", indices[i]);
#endif
		img_subset[i] = images[indices[i]];
		full_img_subset[i] = full_img[indices[i]];
	}
	images = img_subset;
	full_img = full_img_subset;
#if ON_LOGGER
	printf("\n");
#endif
}

void Stitcher::match_pairwise(std::vector<cv::detail::ImageFeatures>& features,
		std::vector<cv::detail::MatchesInfo>& pairwise_matches) {
#if ON_LOGGER
	printf("Match pairwise: ");
#endif
	cv::detail::BestOf2NearestMatcher matcher;
	if (matching_mask.rows * matching_mask.cols <= 1) {
		matcher(features, pairwise_matches);
#if ON_LOGGER
		printf("no matching mask\n");
#endif
	} else {
		matcher(features, pairwise_matches, matching_mask);
#if ON_LOGGER
		printf("use matching mask\n");
#endif
	}
#if ON_DETAIL
	for (auto i : pairwise_matches) {
		if (i.src_img_idx < i.dst_img_idx) {
			printf("	%d %d: %d\n", i.src_img_idx, i.dst_img_idx, i.num_inliers);
		}
	}

#endif
	matcher.collectGarbage();
}

void Stitcher::estimate_camera(std::vector<cv::detail::ImageFeatures>& features,
		std::vector<cv::detail::MatchesInfo>& pairwise_matches,
		std::vector<cv::detail::CameraParams>& cameras) {
	printf("Estimate camera\n");
	cv::detail::HomographyBasedEstimator estimator;
	estimator(features, pairwise_matches, cameras);
#pragma omp parallel for
	for (size_t i = 0; i < cameras.size(); ++i) {
		cv::Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
#if ON_DETAIL
		printf("	Convert camera %ld rotation\n", i);
#endif
	}

}

void Stitcher::refine_camera(
		const std::vector<cv::detail::ImageFeatures>& features,
		const std::vector<cv::detail::MatchesInfo>& pairwise_matches,
		std::vector<cv::detail::CameraParams>& cameras) {
#if ON_LOGGER
	printf("Refine camera\n");
	printf("	Run bundle adjustment\n");
#endif
	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
	adjuster = new cv::detail::BundleAdjusterRay();
	adjuster->setConfThresh(confidence_threshold);
	cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);

	refine_mask(0, 0) = 1;
	refine_mask(0, 1) = 1;
	refine_mask(0, 2) = 1;
	refine_mask(1, 1) = 1;
	refine_mask(1, 2) = 1;

	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);
#if ON_LOGGER
	printf("	Find median focal length: ");
#endif
	// Find median focal length
	std::vector<double> focals(cameras.size());
#pragma omp parallel for
	for (unsigned int i = 0; i < cameras.size(); i++) {
		focals[i] = cameras[i].focal;
	}

	std::sort(focals.begin(), focals.end());
	int focal_end = focals.size();
	while (focal_end > 0) {
		int focal_mid = focals.size() / 2;
		if (focals.size() % 2 == 1) {
			warped_image_scale = static_cast<float>(focals[focal_mid]);
		} else {
			warped_image_scale = static_cast<float>(focals[focal_mid - 1]
					+ focals[focal_mid]) * 0.5f;
		}
		if (isnan(warped_image_scale)) {
			focal_end = focal_mid;
		} else {
			break;
		}
	}

#if ON_LOGGER
	printf("%f\n", warped_image_scale);
#endif
	focals.clear();
#if ON_LOGGER
	printf("	Do wave correction\n");
#endif
	std::vector<cv::Mat> rmats(cameras.size());
#pragma omp parallel for
	for (unsigned int i = 0; i < cameras.size(); i++) {
		rmats[i] = cameras[i].R;
	}
	cv::detail::waveCorrect(rmats, cv::detail::WAVE_CORRECT_HORIZ);
#pragma omp parallel for
	for (unsigned int i = 0; i < cameras.size(); i++) {
		cameras[i].R = rmats[i];
	}
	rmats.clear();
}

void Stitcher::create_warper(cv::Ptr<cv::WarperCreator>& warper_creator) {
#if ON_LOGGER
	printf("Create warper\n");
#endif
	switch (warp_type) {
	case PLANE:
		warper_creator = new cv::PlaneWarper();
		break;
	case CYLINDRICAL:
		warper_creator = new cv::CylindricalWarper();
		break;
	case SPHERICAL:
		warper_creator = new cv::SphericalWarper();
		break;
	case FISHEYE:
		warper_creator = new cv::FisheyeWarper();
		break;
	case STEREOGRAPHIC:
		warper_creator = new cv::StereographicWarper();
		break;
	case COMPRESSEDPLANEA2B1:
		warper_creator = new cv::CompressedRectilinearWarper(2, 1);
		break;
	case COMPRESSEDPLANEA15B1:
		warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
		break;
	case COMPRESSEDPORTRAITA2B1:
		warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
		break;
	case COMPRESSEDPORTRAITA15B1:
		warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
		break;
	case PANINIA2B1:
		warper_creator = new cv::PaniniWarper(2, 1);
		break;
	case PANINIA15B1:
		warper_creator = new cv::PaniniWarper(1.5, 1);
		break;
	case PANINIPORTRAITA2B1:
		warper_creator = new cv::PaniniPortraitWarper(2, 1);
		break;
	case PANINIPORTRAITA15B1:
		warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
		break;
	case MERCATOR:
		warper_creator = new cv::MercatorWarper();
		break;
	case TRANSVERSEMERCATOR:
		warper_creator = new cv::TransverseMercatorWarper();
		break;
	}
}

std::vector<cv::Mat> Stitcher::warp_img(std::vector<cv::Point>& corners,
		const cv::Ptr<cv::WarperCreator>& warper_creator,
		std::vector<cv::Size>& sizes, std::vector<cv::Mat>& masks_warped,
		std::vector<cv::detail::CameraParams>& cameras,
		cv::Ptr<cv::detail::ExposureCompensator>& compensator) {
#if ON_LOGGER
	printf("Warp images\n");
#endif
	std::vector<cv::Mat> masks(num_images);
#pragma omp parallel for
	for (int i = 0; i < num_images; ++i) {
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
	}
	// Warp images and their masks
	std::vector<cv::Mat> images_warped_f(num_images);
	std::vector<cv::Mat> images_warped(num_images);
#pragma omp parallel for
	for (int i = 0; i < num_images; ++i) {
		cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(
				static_cast<float>(warped_image_scale * seam_work_aspect));
		cv::Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float) seam_work_aspect;
		K(0, 0) *= swa;
		K(0, 2) *= swa;
		K(1, 1) *= swa;
		K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR,
				cv::BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST,
				cv::BORDER_CONSTANT, masks_warped[i]);
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
#if ON_DETAIL
		printf("	Warp image and mask %d\n", i);
#endif
	}
#if ON_LOGGER
	printf("Feed exposure compensator\n");
#endif
	compensator = cv::detail::ExposureCompensator::createDefault(
			expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);
	images_warped.clear();
	masks.clear();

	return images_warped_f;

}

void Stitcher::find_seam(std::vector<cv::Mat>& images_warped_f,
		const std::vector<cv::Point>& corners,
		std::vector<cv::Mat>& masks_warped) {
#if ON_LOGGER
	printf("Find seam\n");
#endif
	// Prepare images masks
	cv::Ptr<cv::detail::SeamFinder> seam_finder;
	switch (seam_find_type) {
	case NO:
		seam_finder = new cv::detail::NoSeamFinder();
		break;
	case VORONOI:
		seam_finder = new cv::detail::VoronoiSeamFinder();
		break;
	case GC_COLOR:
		seam_finder = new cv::detail::GraphCutSeamFinder(
				cv::detail::GraphCutSeamFinderBase::COST_COLOR);
		break;
	case GC_COLORGRAD:
		seam_finder = new cv::detail::GraphCutSeamFinder(
				cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
		break;
	case DP_COLOR:
		seam_finder = new cv::detail::DpSeamFinder(
				cv::detail::DpSeamFinder::COLOR);
		break;
	case DP_COLORGRAD:
		seam_finder = new cv::detail::DpSeamFinder(
				cv::detail::DpSeamFinder::COLOR_GRAD);
		break;
	}
	seam_finder->find(images_warped_f, corners, masks_warped);
	// Release unused memory
	images.clear();
}

double Stitcher::resize_mask(const cv::Ptr<cv::WarperCreator>& warper_creator,
		std::vector<cv::Point>& corners, std::vector<cv::Size>& sizes,
		std::vector<cv::detail::CameraParams>& cameras) {
#if ON_LOGGER
	printf("Resize mask\n");
#endif
	double compose_scale = 1;
	double compose_work_aspect = 1;
	if (compositing_resol > 0) {
		compose_scale = std::min(1.0,
				sqrt(compositing_resol * 1e6 / full_img[0].size().area()));
	}

	// Compute relative scales
	compose_work_aspect = compose_scale / work_scale;
	// Update warped image scale
	warped_image_scale *= static_cast<float>(compose_work_aspect);

#pragma omp parallel for
	for (int i = 0; i < num_images; ++i) {
		cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(
				warped_image_scale);

		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx *= compose_work_aspect;
		cameras[i].ppy *= compose_work_aspect;

		// Update corner and size
		cv::Size sz = full_img_sizes;
		if (abs(compose_scale - 1) > 1e-1) {
			sz.width = cvRound(full_img_sizes.width * compose_scale);
			sz.height = cvRound(full_img_sizes.height * compose_scale);
		}

		cv::Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		cv::Rect roi = warper->warpRoi(sz, K, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}
	return compose_scale;

}

cv::Ptr<cv::detail::Blender> Stitcher::prepare_blender(
		const std::vector<cv::Point>& corners,
		const std::vector<cv::Size>& sizes) {
#if ON_LOGGER
	printf("Prepare blender\n");
#endif
	// Update corners and sizes
	cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(
			blend_type, false);
	cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
	if (blend_width < 1.f) {
		blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO,
		false);
	} else {
		if (blend_type == cv::detail::Blender::MULTI_BAND) {
			cv::detail::MultiBandBlender* mb =
					dynamic_cast<cv::detail::MultiBandBlender*>(static_cast<cv::detail::Blender*>(blender));
			mb->setNumBands(
					static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
#if ON_LOGGER
			printf("	Number of bands: %d\n", mb->numBands());
#endif
		} else {
			if (blend_type == cv::detail::Blender::FEATHER) {
				cv::detail::FeatherBlender* fb =
						dynamic_cast<cv::detail::FeatherBlender*>(static_cast<cv::detail::Blender*>(blender));
				fb->setSharpness(1.f / blend_width);
#if ON_LOGGER
				printf("	Sharpness: %f\n", fb->sharpness());
#endif
			}
		}
	}
	blender->prepare(corners, sizes);

	return blender;
}

void Stitcher::blend_img(const double& compose_scale,
		const cv::Ptr<cv::WarperCreator>& warper_creator,
		cv::Ptr<cv::detail::ExposureCompensator>& compensator,
		cv::vector<cv::Point>& corners, std::vector<cv::Mat>& masks_warped,
		cv::Ptr<cv::detail::Blender>& blender,
		std::vector<cv::detail::CameraParams>& cameras, cv::Mat& result) {
#if ON_LOGGER
	printf("Blend pano\n");
#endif
	img.resize(num_images);
#pragma omp parallel for
	for (int img_idx = 0; img_idx < num_images; ++img_idx) {
#if ON_DETAIL
		printf("	Resize image\n");
#endif
		// Read image and resize it if necessary
		if (abs(compose_scale - 1) > 1e-1) {
			cv::resize(full_img[img_idx], img[img_idx], cv::Size(),
					compose_scale, compose_scale);
		} else {
			img[img_idx] = full_img[img_idx];
		}
		cv::Size img_size = img[img_idx].size();

		cv::Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);
#if ON_DETAIL
		printf("	Warp image\n");
#endif
		// Warp the current image
		cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(
				warped_image_scale);
		cv::Mat img_warped;
		warper->warp(img[img_idx], K, cameras[img_idx].R, cv::INTER_LINEAR,
				cv::BORDER_REFLECT, img_warped);

#if ON_DETAIL
		printf("	Warp mask\n");
#endif
		// Warp the current image mask
		cv::Mat mask;
		mask.create(img_size, CV_8U);
		mask.setTo(cv::Scalar::all(255));
		cv::Mat mask_warped;
		warper->warp(mask, K, cameras[img_idx].R, cv::INTER_NEAREST,
				cv::BORDER_CONSTANT, mask_warped);
		mask.release();
#if ON_DETAIL
		printf("	Compensate exposure\n");
#endif
		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
		cv::Mat img_warped_s;
		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		cv::Mat dilated_mask;
		dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
		cv::Mat seam_mask;
		cv::resize(dilated_mask, seam_mask, mask_warped.size());
		dilated_mask.release();
		mask_warped = seam_mask & mask_warped;
		seam_mask.release();
		// Blend the current image
#if ON_DETAIL
		printf("	Image %d feeded\n", img_idx);
#endif
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		mask_warped.release();
		img_warped_s.release();
	}
	cv::Mat result_mask;
	blender->blend(result, result_mask);
}

int Stitcher::registration(std::vector<cv::detail::CameraParams>& cameras) {
#if ON_LOGGER
	long long start;
	printf("=========================================================\n");
	printf("Registration stage\n");
#endif
	int retVal = 1; //1 is normal, 0 is not enough, -1 is failed
	img.resize(num_images);
	images.resize(num_images);

	cv::vector<cv::detail::ImageFeatures> features(num_images);
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	find_features(features);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	cv::vector<cv::detail::MatchesInfo> pairwise_matches;
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	match_pairwise(features, pairwise_matches);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

#if ON_LOGGER
	start = cv::getTickCount();
#endif
	// Leave only images we are sure are from the same panorama
	extract_biggest_component(features, pairwise_matches);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	// Check if we still have enough images
	int tmp = static_cast<int>(images.size());
	status.second = double(tmp) / num_images;
	if (tmp < 2) {
		return -1;
	}
	if (tmp < num_images) {
		num_images = tmp;
		retVal = 0;
	}
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	estimate_camera(features, pairwise_matches, cameras);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

#if ON_LOGGER
	start = cv::getTickCount();
#endif
	refine_camera(features, pairwise_matches, cameras);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif
	features.clear();
	pairwise_matches.clear();
	return retVal;
}

cv::Mat Stitcher::compositing(std::vector<cv::detail::CameraParams>& cameras) {
#if ON_LOGGER
	long long start;
	printf("=========================================================\n");
	printf("Compositing\n");
#endif
	cv::Ptr<cv::WarperCreator> warper_creator;

	// Warp images and their masks
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	create_warper(warper_creator);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	std::vector<cv::Point> corners(num_images);
	std::vector<cv::Mat> masks_warped(num_images);
	cv::Ptr<cv::detail::ExposureCompensator> compensator;
	std::vector<cv::Size> sizes(num_images);

#if ON_LOGGER
	start = cv::getTickCount();
#endif
	std::vector<cv::Mat> images_warped_f = warp_img(corners, warper_creator,
			sizes, masks_warped, cameras, compensator);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	// Prepare images masks
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	find_seam(images_warped_f, corners, masks_warped);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif
	images_warped_f.clear();

#if ON_LOGGER
	start = cv::getTickCount();
#endif
	double compose_scale = resize_mask(warper_creator, corners, sizes, cameras);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	// Update corners and sizes
#if ON_LOGGER
	start = cv::getTickCount();
#endif
	cv::Ptr<cv::detail::Blender> blender = prepare_blender(corners, sizes);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif
	cv::Mat result;

#if ON_LOGGER
	start = cv::getTickCount();
#endif
	blend_img(compose_scale, warper_creator, compensator, corners, masks_warped,
			blender, cameras, result);
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif

	corners.clear();
	masks_warped.clear();
	sizes.clear();

	return result;
}

Stitcher::Stitcher() {
#if ON_LOGGER
	printf("Create stitcher using no argument\n");
#endif
	init(FAST);
}

void Stitcher::init(const Stitcher::InitMode& mode) {
	warped_image_scale = 1.0;
	num_images = full_img.size();
	blend_type = cv::detail::Blender::MULTI_BAND;
	seam_work_aspect = 1.0;
	work_scale = 1.0;
	warp_type = CYLINDRICAL;
	seam_find_type = DP_COLORGRAD;
	expos_comp_type = cv::detail::ExposureCompensator::GAIN;
	switch (mode) {
	case NORMAL:
		registration_resol = 0.6;
		break;
	case FAST:
		registration_resol = 0.3;
		break;
	}
	confidence_threshold = 1.0;
	seam_estimation_resol = 0.08;
	compositing_resol = -1.0;
	matching_mask = cv::Mat(1, 1, CV_8U, cv::Scalar(0));
	status = {OK, -1};
}

void Stitcher::set_matching_mask(const std::string& file_name,
		std::vector<std::pair<int, int> >& edge_list) {
	struct stat buf;
	if (stat(file_name.c_str(), &buf) != -1) {
#if ON_LOGGER
		printf("	Input matching mask from file\n");
#endif
		std::ifstream pairwise(file_name.c_str(), std::ifstream::in);
		while (true) {
			int i, j;
			pairwise >> i >> j;
			std::pair<int, int> t;
			t = {i,j};
			edge_list.push_back(t);
			if (pairwise.eof()) {
				break;
			}
		}
		pairwise.close();
	}
}

int Stitcher::rotate_img(const std::string& img_path) {
#if ON_LOGGER
	printf("	Rotate images if necessary: ");
#endif
	try {
		Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(img_path);
		assert(image.get() != 0);
		image->readMetadata();
		Exiv2::ExifData &exifData = image->exifData();
		if (exifData.empty()) {
			throw Exiv2::Error(1, "No exif data found!");
		}
		Exiv2::ExifData::const_iterator i = exifData.findKey(
				Exiv2::ExifKey("Exif.Image.Orientation"));
		if (i == exifData.end()) {
			throw Exiv2::Error(2, "Orientation not found!");
		}
		int angle = i->value().toLong();
		switch (angle) {
		case 8: {
#if ON_LOGGER
			printf("pi/4 radian CW");
#endif
#pragma omp parallel for
			for (int i = 0; i < num_images; i++) {
				transpose(full_img[i], full_img[i]);
				flip(full_img[i], full_img[i], 0);
			}
		}
			break;
		case 6: {
#if ON_LOGGER
			printf("pi/4 radian CCW");
#endif
#pragma omp parallel for
			for (int i = 0; i < num_images; i++) {
				transpose(full_img[i], full_img[i]);
				flip(full_img[i], full_img[i], 1);
			}
		}
			break;
		case 3: {
#if ON_LOGGER
			printf("pi/2 radian CCW");
#endif
#pragma omp parallel for
			for (int i = 0; i < num_images; i++) {
				flip(full_img[i], full_img[i], -1);
			}
			break;
		}
		case 1:
#if ON_LOGGER
			printf("no rotation.");
#endif
			break;
		}
	} catch (Exiv2::AnyError& e) {
#if ON_LOGGER
		printf(" %s\n", e.what());
#endif
		return -1;
	}
	printf("\n");
	return 0;
}

/*void Stitcher::feed(const std::string& dir)
 {
 #if ON_LOGGER
 printf("Scan directory to find input images and matching masks\n");
 #endif
 boost::filesystem::path dir_path(dir);
 std::string supported_format = ".jpg .jpeg .jpe .jp2 .png .bmp .dib .tif .tiff .pbm .pgm .ppm .sr .ras";
 try
 {
 if (boost::filesystem::exists(dir_path) && boost::filesystem::is_directory(dir_path))
 {
 std::vector<std::string> img_name;
 std::vector<std::pair<int, int> > edge_list;
 boost::filesystem::directory_iterator it(dir);
 while (it != boost::filesystem::directory_iterator())
 {
 boost::filesystem::path file_path = it->path();

 if ((boost::filesystem::is_regular_file(file_path)))
 {
 std::string extension = file_path.extension().c_str();
 std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
 if (supported_format.find(extension) != std::string::npos)
 {
 img_name.push_back(file_path.c_str());
 num_images++;
 }
 else
 {
 if (file_path.filename().compare("pairwise.txt") == 0)
 {
 set_matching_mask(file_path.c_str(), edge_list);
 }
 }
 }
 it++;
 }

 std::sort(img_name.begin(), img_name.end());
 full_img.resize(num_images);
 std::vector<cv::Size> full_img_tmp_size(num_images);
 #pragma omp parallel for
 for (int i = 0; i < num_images; i++)
 {
 full_img[i] = cv::imread(img_name[i]);
 full_img_tmp_size[i] = full_img[i].size();
 }

 std::sort(full_img_tmp_size.begin(), full_img_tmp_size.end(), compareCvSize);
 full_img_sizes = full_img_tmp_size[0];
 if (compareCvSize(full_img_tmp_size[0], full_img_tmp_size[num_images - 1]))
 {
 #pragma omp parallel for
 for (int i = 0; i < num_images; i++)
 {
 cv::resize(full_img[i], full_img[i], full_img_sizes);
 }
 }
 full_img_tmp_size.clear();

 rotate_img(img_name[0]);
 full_img_sizes = full_img[0].size();
 img_name.clear();

 if (!edge_list.empty())
 {
 matching_mask = cv::Mat(num_images, num_images, CV_8U, cv::Scalar(0));
 #pragma omp parallel for
 for (unsigned int i = 0; i < edge_list.size(); i++)
 {
 int t_first = edge_list[i].first, t_second =
 edge_list[i].second;
 if (t_first >= 0 && t_first < num_images && t_second >= 0 && t_second < num_images)
 {
 matching_mask.at<char>(t_first, t_second) = 1;
 }
 }
 }
 edge_list.clear();
 }
 }
 catch (const boost::filesystem::filesystem_error& ex)
 {
 #if ON_LOGGER
 printf("Error when processing files.\n");
 #endif
 return;
 }
 }*/

void Stitcher::feed(const std::string& input_dir) {
	num_images = 0;
	struct stat buf;
	std::string pairwise_path = input_dir + "pairwise.txt";
	std::vector<std::string> img_name;
	std::vector<std::pair<int, int>> pairwise;
	if (stat(pairwise_path.c_str(), &buf) != -1) {
#if ON_LOGGER
		printf("Input from pairwise.txt\n");
#endif
		std::string src_img, dst_img;
		std::ifstream ifs(pairwise_path, std::ifstream::in);
		int src_idx = 0, dst_idx = 0;
		while (!ifs.eof()) {
			ifs >> src_img >> dst_img;
			unsigned int i = 0;
			for (i = 0; i < img_name.size(); i++) {
				if (img_name[i].substr(img_name[i].find_last_of('/')+1) == src_img) {
					src_idx = i;
					break;
				}
			}
			if (i == img_name.size()) {
				src_idx = img_name.size();
				img_name.push_back(input_dir + src_img);
			}
			i = 0;
			for (i = 0; i < img_name.size(); i++) {
				if (img_name[i].substr(img_name[i].find_last_of('/')+1) == dst_img) {
					dst_idx = i;
					break;
				}
			}
			if (i == img_name.size()) {
				dst_idx = img_name.size();
				img_name.push_back(input_dir + dst_img);
			}
			if (src_idx > dst_idx)
				std::swap(src_idx, dst_idx);
#if ON_LOGGER
		printf("%d %d\n", src_idx, dst_idx);
#endif
			pairwise.push_back(std::make_pair(src_idx, dst_idx));
		}
		ifs.close();
	} else {
#if ON_LOGGER
		printf("Scan directory to find input images\n");
#endif
		boost::filesystem::path dir_path(input_dir);
		std::string supported_format =
				".jpg .jpeg .jpe .jp2 .png .bmp .dib .tif .tiff .pbm .pgm .ppm .sr .ras";
		try {
			if (boost::filesystem::exists(dir_path)
					&& boost::filesystem::is_directory(dir_path)) {
				boost::filesystem::directory_iterator it(input_dir);
				while (it != boost::filesystem::directory_iterator()) {
					boost::filesystem::path file_path = it->path();
					if ((boost::filesystem::is_regular_file(file_path))) {
						std::string extension = file_path.extension().c_str();
						std::transform(extension.begin(), extension.end(),
								extension.begin(), ::tolower);
						if (supported_format.find(extension)
								!= std::string::npos) {
							img_name.push_back(file_path.c_str());
							num_images++;
						}
					}
					it++;
				}
				std::sort(img_name.begin(), img_name.end());
			}
		} catch (const boost::filesystem::filesystem_error& ex) {
#if ON_LOGGER
			printf("%s\n", ex.what());
#endif
		}
	}
	//Resize and rotate all images to a fixed sizes
	num_images = img_name.size();
	if (num_images < 2)
		return;
	full_img.resize(num_images);
	std::vector<cv::Size> full_img_tmp_size(num_images);
#pragma omp parallel for
	for (int i = 0; i < num_images; i++) {
		full_img[i] = cv::imread(img_name[i]);
		full_img_tmp_size[i] = full_img[i].size();
	}
	sort(full_img_tmp_size.begin(), full_img_tmp_size.end(), compareCvSize);
	full_img_sizes = full_img_tmp_size[0];
	if (compareCvSize(full_img_tmp_size[0],
			full_img_tmp_size[num_images - 1])) {
#pragma omp parallel for
		for (int i = 0; i < num_images; i++) {
			resize(full_img[i], full_img[i], full_img_sizes);
		}
	}
	full_img_tmp_size.clear();
	rotate_img(img_name[0]);
	full_img_sizes = full_img[0].size();
#if ON_LOGGER
	printf("	Input sizes: %dx%d\n", full_img_sizes.height,
			full_img_sizes.width);
#endif
	//Set matching mask
	if (!pairwise.empty()) {
		matching_mask = cv::Mat(num_images, num_images, CV_8U, cv::Scalar(0));
#pragma omp parallel for
		for (unsigned int i = 0; i < pairwise.size(); i++) {
			matching_mask.at<unsigned char>(pairwise[i].first,
					pairwise[i].second) = 1;
		}
	}
}

void Stitcher::set_dst(const std::string &dst) {
	result_dst = dst;
}

std::string Stitcher::get_dst() {
	return result_dst;
}

void Stitcher::stitching_process(cv::Mat& result) {
	enum ReturnCode retVal = OK;
	if (full_img.size() < 2) {
		retVal = NEED_MORE;
	} else {
		cv::vector<cv::detail::CameraParams> cameras;
		int check = registration(cameras);
		if (check == -1) {
			retVal = FAILED;
		} else {
			if (check == 0) {
				retVal = NOT_ENOUGH;
			}
			result = compositing(cameras);
			if (result.rows * result.cols == 1) {
				retVal = FAILED;
			}
		}
		cameras.clear();
	}

	status.first = retVal;
}

void Stitcher::collect_garbage() {
	full_img.clear();
	img.clear();
	images.clear();
	//full_img_sizes.clear();
}

void Stitcher::stitch() {
	cv::Mat result;
	std::vector<cv::Mat> img_bak = full_img;
#if ON_LOGGER
	printf("1st try\n");
#endif
	stitching_process(result);
	std::pair<ReturnCode, double> tmp_code = status;
#if ON_LOGGER
	printf("%d %lf\n\n", status.first, status.second);
#endif
	if (status.first == NEED_MORE) {
		return;
	}
	if (status.first != OK) {
		cv::Mat retry;
		collect_garbage();
		init(NORMAL);
		full_img = img_bak;
		num_images = full_img.size();
#if ON_LOGGER
		printf("2nd try\n");
#endif
		stitching_process(retry);
#if ON_LOGGER
		printf("%d %lf\n", status.first, status.second);
#endif
		switch (status.first) {
		case NEED_MORE:
			return;
		case OK:
			result = retry.clone();
			break;
		case NOT_ENOUGH:
			if (tmp_code.first == status.first
					&& tmp_code.second < status.second) {
				result = retry.clone();
			} else {
				status = tmp_code;
			}
			break;
		case FAILED:
			status = tmp_code;
			break;
		}
	}
#if ON_LOGGER
	printf("Write final pano ");
	long long start = cv::getTickCount();
#endif
#pragma omp parallel sections
	{
		{
			std::string tmp_result = result_dst + ".jpg";
			cv::imwrite(tmp_result, result);
		}
#pragma omp section
		{
			double scale = double(1080) / result.rows;
			cv::Mat preview;
			if (scale < 1.25f) {
				cv::resize(result, preview, cv::Size(), scale, scale);
			} else {
				preview = result.clone();
			}
			std::vector<int> compression_para;
			compression_para.push_back(CV_IMWRITE_JPEG_QUALITY);
			compression_para.push_back(75);
			std::string tmp_preview = result_dst + "p.jpg";
			cv::imwrite(tmp_preview, preview, compression_para);
		}
	}
#if ON_LOGGER
	printf("%lf\n",
			(double(cv::getTickCount()) - start) / cv::getTickFrequency());
#endif
}

std::string Stitcher::get_status() {
	std::ostringstream ostr;
	switch (status.first) {
	case OK:
		ostr << "OK " << status.second;
		break;
	case NOT_ENOUGH:
		ostr << "Not enough " << status.second;
		break;
	case FAILED:
		return "Failed";
	case NEED_MORE:
		return "Need more images";
	}
	return ostr.str();
}

Stitcher::~Stitcher() {

}

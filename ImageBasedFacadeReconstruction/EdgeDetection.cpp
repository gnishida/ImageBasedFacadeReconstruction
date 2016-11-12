#include "EdgeDetection.h"
#include "HoughTransform.h"
#include "CVUtils.h"

namespace ed {
	
	const double M_PI = 3.1415926535897932384626433832795;

	bool order_votes(const std::pair<float, std::pair<float, float>>& left, const std::pair<float, std::pair<float, float>>& right) {
		return left.first > right.first;
	}

	/**
	 * Detect horizontal and vertical dominant orientations.
	 *
	 * @param img					image
	 * @param angle_threshold		threshold for horizontal and vertical angles
	 * @param voteThresholdRatio	threshold ratio for the Hough accumulator
	 * @return						horizontal and vertical angles in degree
	 */
	std::pair<float, float> detectHorizontalAndVerticalDominantOrientation(const cv::Mat& img, float angle_threshold, bool remove_diagonal_edges, bool use_magnitude_as_weight, bool smooth_accum, float voteThresholdRatio) {
		// convert to grayscale
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		// blur the image
		//cv::blur(grayImg, grayImg, cv::Size(5, 5));

		// Canny edge to get an edge image
		cv::Mat edges;
		//cv::Canny(grayImg, edges, 50, 120);
		autoCanny2(grayImg, edges);

		if (remove_diagonal_edges) {
			removeDiagonalEdges(grayImg, edges, angle_threshold);
		}

		// compute gradient magnitude
		cv::Mat grad;
		gradientMagnitude(grayImg, grad);

		// apply Hough transform
		cv::Mat accum;
		HoughTransform::computeAccumulator(edges, accum, use_magnitude_as_weight, grad);

		if (smooth_accum) {
			cv::blur(accum, accum, cv::Size(5, 5));
		}

		// supress non-maximam votes
		HoughTransform::suppressNonMaximum(accum, accum, 6);

		float v_max = getVerticalAccumMax(accum, angle_threshold);
		float h_max = getHorizontalAccumMax(accum, angle_threshold);

		// threshold
		for (int r = 0; r < accum.rows; ++r) {
			for (int t = 0; t < accum.cols; ++t) {
				if (t <= angle_threshold || t >= 180 - angle_threshold) {
					if (accum.at<float>(r, t) < v_max * voteThresholdRatio) {
						accum.at<float>(r, t) = 0;
					}
				}
				else if (t >= 90 - angle_threshold && t <= 90 + angle_threshold) {
					if (accum.at<float>(r, t) < h_max * voteThresholdRatio) {
						accum.at<float>(r, t) = 0;
					}
				}
				else {
					accum.at<float>(r, t) = 0;
				}
			}
		}

		// sum up the votes in terms of r
		cv::reduce(accum, accum, 0, CV_REDUCE_SUM);

		float vert = 90.0f;
		float hori = 0.0f;
		v_max = 0.0f;
		h_max = 0.0f;
		for (int t = 0; t < accum.cols; ++t) {
			if (t <= angle_threshold || t >= 180 - angle_threshold) {
				if (accum.at<float>(0, t) > v_max) {
					v_max = accum.at<float>(0, t);
					vert = t + 90;
					if (vert > 180) vert -= 180;
				}
			}
			else if (t >= 90 - angle_threshold && t <= 90 + angle_threshold) {
				if (accum.at<float>(0, t) > h_max) {
					h_max = accum.at<float>(0, t);
					hori = t - 90;
				}
			}
		}

		return std::make_pair(hori, vert);
	}

	/**
	 * Detect horizontal and vertical edges and return the list of the pair of coordinates of the end points of edges.
	 *
	 * @param img				image
	 * @param angle_threshold	threshold for horizontal and vertical angles
	 * @param edge_detector		type of edge detector (0 - cv::Canny / 1 - autoCanny / 2 - autoCanny2)
	 * @param smooth_accum		smooth the accumulator if true
	 * @param topN				top N to return
	 */
	std::vector<std::tuple<glm::vec2, glm::vec2, int, float>> detectHorizontalAndVerticalEdges(const cv::Mat& img, float angle_threshold, int edge_detector, bool remove_diagonal_edges, bool use_magnitude_as_weight, bool smooth_accum, int topN) {
		// convert to grayscale
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		// blur the image
		cv::blur(grayImg, grayImg, cv::Size(5, 5));

		// Canny edge to get an edge image
		cv::Mat edges;
		if (edge_detector == 0) {
			cv::Canny(grayImg, edges, 50, 120);
		}
		else if (edge_detector == 1) {
			autoCanny(grayImg, edges);
		}
		else if (edge_detector == 2) {
			autoCanny2(grayImg, edges);
		}

		if (remove_diagonal_edges) {
			removeDiagonalEdges(grayImg, edges, angle_threshold);
		}

		// compute gradient magnitude
		cv::Mat grad;
		gradientMagnitude(grayImg, grad);
		
		// apply Hough transform
		cv::Mat accum;
		HoughTransform::computeAccumulator(edges, accum, use_magnitude_as_weight, grad);

		if (smooth_accum) {
			cv::blur(accum, accum, cv::Size(5, 5));
		}

		// supress non-maximam votes
		HoughTransform::suppressNonMaximum(accum, accum, 6);

		// build vectors for sorting later
		cv::Mat accum_suppressed = accum.clone();
		std::vector<std::pair<float, std::pair<float, float>>> hor_ordered_votes;
		std::vector<std::pair<float, std::pair<float, float>>> ver_ordered_votes;
		for (int r = 0; r < accum.rows; ++r) {
			for (int t = 0; t < accum.cols; ++t) {
				if (accum.at<float>(r, t) > 0) {
					if (t <= angle_threshold || t >= 180 - angle_threshold) {
						ver_ordered_votes.push_back(std::make_pair(accum.at<float>(r, t), std::make_pair(r - accum.rows / 2, t)));
					}
					else if (t >= 90 - angle_threshold && t <= 90 + angle_threshold) {
						hor_ordered_votes.push_back(std::make_pair(accum.at<float>(r, t), std::make_pair(r - accum.rows / 2, t)));
					}
				}
			}
		}

		// sort the edges by the number of votes
		std::sort(hor_ordered_votes.begin(), hor_ordered_votes.end(), order_votes);
		std::sort(ver_ordered_votes.begin(), ver_ordered_votes.end(), order_votes);

		std::vector<std::tuple<glm::vec2, glm::vec2, int, float>> detected_edges;
		for (int i = 0; i < std::min(topN, (int)std::max(hor_ordered_votes.size(), ver_ordered_votes.size())); ++i) {
			if (i < hor_ordered_votes.size()) {
				float votes = hor_ordered_votes[i].first;
				float r = hor_ordered_votes[i].second.first;
				float t = hor_ordered_votes[i].second.second;

				glm::vec2 p1, p2;
				p1.x = 0;
				p1.y = (r + img.cols / 2 * cosf(t / 180.0 * M_PI)) / sinf(t / 180.0f * M_PI) + img.rows / 2;
				p2.x = img.cols - 1;
				p2.y = (r - img.cols / 2 * cosf(t / 180.0 * M_PI)) / sinf(t / 180.0f * M_PI) + img.rows / 2;

				detected_edges.push_back(std::make_tuple(p1, p2, 0, votes));
			}

			if (i < ver_ordered_votes.size()) {
				float votes = ver_ordered_votes[i].first;
				float r = ver_ordered_votes[i].second.first;
				float t = ver_ordered_votes[i].second.second;

				glm::vec2 p1, p2;
				p1.x = (r + img.rows / 2 * sinf(t / 180.0 * M_PI)) / cosf(t / 180.0f * M_PI) + img.cols / 2;
				p1.y = 0;
				p2.x = (r - img.rows / 2 * sinf(t / 180.0 * M_PI)) / cosf(t / 180.0f * M_PI) + img.cols / 2;
				p2.y = img.rows - 1;

				detected_edges.push_back(std::make_tuple(p1, p2, 1, votes));
			}
		}

		return detected_edges;
	}

	void removeDiagonalEdges(const cv::Mat& grayImg, cv::Mat& edgeImg, float angle_threshold) {
		// compute the gradient orientation
		cv::Mat gradOri;
		ed::gradientOrientation(grayImg, gradOri);

		// keep only horizontal and vertical edges
		for (int r = 0; r < edgeImg.rows; ++r) {
			for (int c = 0; c < edgeImg.cols; ++c) {
				// if the gradient orientation is neither horizontal nor vertical, remove the edge
				if ((gradOri.at<float>(r, c) > angle_threshold && gradOri.at<float>(r, c) < 90 - angle_threshold)
					|| (gradOri.at<float>(r, c) > 90 + angle_threshold && gradOri.at<float>(r, c) < 180 - angle_threshold)) {
					edgeImg.at<uchar>(r, c) = 0;
				}
			}
		}
	}

	/**
	 * Warp the image based on the horizontal and vertical orientations.
	 *
	 * @param img		image
	 * @param hori		horizontal orientation in degree
	 * @param vert		vertical orientation in degree
	 * @param warped	warped image
	 */
	void warp(cv::Mat img, float hori, float vert, cv::Mat& warped) {
		std::vector<cv::Point2f> srcTri(3);
		std::vector<cv::Point2f> dstTri(3);
		srcTri[0] = cv::Point2f(img.cols / 2.0, img.rows / 2.0);
		srcTri[1] = cv::Point2f(img.cols, img.rows / 2.0 + img.cols / 2.0 * tan(hori / 180.0 * M_PI));
		srcTri[2] = cv::Point2f(img.cols / 2.0 + img.rows / 2.0 / tan(vert / 180.0 * M_PI), img.rows);
		dstTri[0] = cv::Point2f(img.cols / 2.0, img.rows / 2.0);
		dstTri[1] = cv::Point2f(img.cols, img.rows / 2.0);
		dstTri[2] = cv::Point2f(img.cols / 2.0, img.rows);
		cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
		
		cv::warpAffine(img, warped, warpMat, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	}

	/**
	 * Compute the gradient orientation at each pixel in degree.
	 *
	 * @param img		image
	 * @param gradOri	gradient orientation
	 */
	void gradientOrientation(cv::Mat img, cv::Mat& gradOri) {
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		cv::Mat sobelx;
		cv::Mat sobely;
		cv::Sobel(grayImg, sobelx, CV_32F, 1, 0, 3);
		cv::Sobel(grayImg, sobely, CV_32F, 0, 1, 3);

		gradOri = cv::Mat(grayImg.size(), CV_32F);
		for (int r = 0; r < sobelx.rows; ++r) {
			for (int c = 0; c < sobelx.cols; ++c) {
				float theta = atan2f(sobely.at<float>(r, c), sobelx.at<float>(r, c));
				float deg = theta / M_PI * 180.0f;
				if (deg < 0) deg += 180;
				if (deg >= 180) deg -= 180;

				gradOri.at<float>(r, c) = deg;
			}
		}
	}

	void gradientMagnitude(cv::Mat img, cv::Mat& grad) {
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		cv::Mat sobelx;
		cv::Mat sobely;
		cv::Sobel(grayImg, sobelx, CV_32F, 1, 0, 3);
		cv::Sobel(grayImg, sobely, CV_32F, 0, 1, 3);

		grad = cv::Mat(grayImg.size(), CV_32F);
		for (int r = 0; r < sobelx.rows; ++r) {
			for (int c = 0; c < sobelx.cols; ++c) {
				grad.at<float>(r, c) = sqrtf(sobely.at<float>(r, c) * sobely.at<float>(r, c) + sobelx.at<float>(r, c) * sobelx.at<float>(r, c));
			}
		}
	}

	void autoCanny(const cv::Mat& grayImg, cv::Mat& edgeImg, float sigma) {
		float v = median(grayImg);
		int lower = std::max(0.0f, (1.0f - sigma) * v);
		int upper = std::max(255.0f, (1.0f + sigma) * v);

		cv::Canny(grayImg, edgeImg, lower, upper);
	}

	void autoCanny2(const cv::Mat& grayImg, cv::Mat& edgeImg) {
		cv::Mat sobelx;
		cv::Sobel(grayImg, sobelx, CV_64F, 1, 0);
		cv::Mat sobely;
		cv::Sobel(grayImg, sobely, CV_64F, 0, 1);
		cv::Mat grad = (cv::abs(sobelx) + cv::abs(sobely)) * 0.5;
		grad.convertTo(grad, CV_8U);
		

		float v = median(grad) * 2;

		cv::Canny(grayImg, edgeImg, v * 0.6, v * 1.2);
	}

	double median(const cv::Mat& grayImg) {
		double m = (grayImg.rows * grayImg.cols) / 2;
		int bin = 0;
		double med = -1.0;

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		cv::Mat hist;
		cv::calcHist(&grayImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		for (int i = 0; i < histSize && med < 0.0; ++i) {
			bin += cvRound(hist.at< float >(i));
			if (bin > m && med < 0.0)
				med = i;
		}

		return med;
	}

	float getVerticalAccumMax(const cv::Mat& accum, float max_degree) {
		cv::Mat accum_max;
		cv::reduce(accum, accum_max, 0, CV_REDUCE_MAX);

		// find the maximum value
		float v_max = 0.0f;
		for (int t = 0; t < 180; ++t) {
			if (t > max_degree && t < 180 - max_degree) continue;

			if (accum_max.at<float>(0, t) > v_max) {
				v_max = accum_max.at<float>(0, t);
			}
		}

		return v_max;
	}

	float getHorizontalAccumMax(const cv::Mat& accum, float max_degree) {
		cv::Mat accum_max;
		cv::reduce(accum, accum_max, 0, CV_REDUCE_MAX);

		// find the maximum value
		float h_max = 0.0f;
		for (int t = 90 - max_degree; t <= 90 + max_degree; ++t) {
			if (accum_max.at<float>(0, t) > h_max) {
				h_max = accum_max.at<float>(0, t);
			}
		}

		return h_max;
	}


}


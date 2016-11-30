#include "FacadeSegmentation.h"
#include "CVUtils.h"
#include "Utils.h"
#include <fstream>

namespace fs {

	void subdivideFacade(cv::Mat img, int num_floors, bool align_windows, std::vector<float>& y_splits, std::vector<float>& x_splits, std::vector<std::vector<WindowPos>>& win_rects) {
		// gray scale
		cv::Mat gray_img;
		cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
		
		// average floor height
		float average_floor_height = (float)img.rows / num_floors;

		// compute kernel size
		int kernel_size = average_floor_height / 6;
		if (kernel_size % 2 == 0) kernel_size++;

		// blur the image according to the average floor height
		cv::Mat blurred_gray_img;
		if (kernel_size > 1) {
			cv::GaussianBlur(gray_img, blurred_gray_img, cv::Size(kernel_size, kernel_size), kernel_size);
		}
		else {
			blurred_gray_img = gray_img.clone();
		}

		// compute Ver and Hor
		cv::Mat_<float> Ver, Hor;
		computeVerAndHor2(blurred_gray_img, Ver, Hor);

		// smooth Ver and Hor
		if (kernel_size > 1) {
			cv::blur(Ver, Ver, cv::Size(kernel_size, kernel_size));
			cv::blur(Hor, Hor, cv::Size(kernel_size, kernel_size));
		}


		////////////////////////////////////////////////////////////////////////////////////////////////
		// subdivide vertically
		
		cv::Range h_range = cv::Range(average_floor_height * 0.8, average_floor_height * 1.5);

		// find the local minima of Ver
		std::vector<float> y_splits_strong;
		getSplitLines(Ver, 0.3, y_splits_strong);
		getSplitLines(Ver, 0.15, y_splits);

		// compute S_max and h_max
		std::map<float, int> SV_y_map;
		std::map<int, float> SV_max;
		std::map<int, int> h_max;
		std::map<int, bool> y_strong_splits;
		float tau_max = 0;
		for (int i = 0; i < y_splits.size(); ++i) {
			float SV;
			int h;
			computeSV(blurred_gray_img, y_splits[i], SV, h, h_range);

			SV_y_map[-SV] = y_splits[i]; // to sort Y by SV in a descending order
			SV_max[y_splits[i]] = SV;
			h_max[y_splits[i]] = h;

			// check whether the split at this point is a strong one or not.
			if (std::find(y_splits_strong.begin(), y_splits_strong.end(), y_splits[i]) != y_splits_strong.end()) {
				y_strong_splits[y_splits[i]] = true;
			}
			else {
				y_strong_splits[y_splits[i]] = false;
			}

			// update tau_max
			if (SV > tau_max) {
				tau_max = SV;
			}
		}

		// sort y_candidates by SV in a descending order
		std::vector<int> y_candidates;
		for (auto it = SV_y_map.begin(); it != SV_y_map.end(); ++it) {
			int y = it->second;
			y_candidates.push_back(y);
		}
		
		// find the floor boundaries
		std::vector<std::vector<std::pair<int, int>>> symmetry_listV;
		y_splits = findSymmetryV(blurred_gray_img, h_range, SV_max, h_max, y_candidates, y_strong_splits, tau_max, cv::Range(0, blurred_gray_img.rows - 1), symmetry_listV);
		std::sort(y_splits.begin(), y_splits.end());

		if (y_splits.size() == 0 || y_splits[0] > 0) {
			y_splits.insert(y_splits.begin(), 0);
		}
		if (y_splits.back() < blurred_gray_img.rows - 1) {
			y_splits.push_back(blurred_gray_img.rows - 1);
		}
		

		////////////////////////////////////////////////////////////////////////////////////////////////
		// subdivide horizontally

		cv::Range w_range = cv::Range(average_floor_height * 0.7, average_floor_height * 2.4);

		// find the local minima of Hor
		std::vector<float> x_splits_strong;
		getSplitLines(Hor, 0.3, x_splits_strong);
		getSplitLines(Hor, 0.15, x_splits);

		// compute S_max and w_max
		std::map<float, int> SH_x_map;
		std::map<int, float> SH_max;
		std::map<int, int> w_max;
		std::map<int, bool> x_strong_splits;
		tau_max = 0;
		for (int i = 0; i < x_splits.size(); ++i) {
			float SH;
			int w;
			computeSH(blurred_gray_img, x_splits[i], SH, w, w_range);

			SH_x_map[-SH] = x_splits[i]; // to sort X by SV in a descending order
			SH_max[x_splits[i]] = SH;
			w_max[x_splits[i]] = w;

			// check whether the split at this point is a strong one or not.
			if (std::find(x_splits_strong.begin(), x_splits_strong.end(), x_splits[i]) != x_splits_strong.end()) {
				x_strong_splits[x_splits[i]] = true;
			}
			else {
				x_strong_splits[x_splits[i]] = false;
			}

			// update tau_max
			if (SH > tau_max) {
				tau_max = SH;
			}
		}

		// sort x_candidates by SH in a descending order
		std::vector<int> x_candidates;
		for (auto it = SH_x_map.begin(); it != SH_x_map.end(); ++it) {
			int x = it->second;
			x_candidates.push_back(x);
		}

		// find the floor boundaries
		std::vector<std::vector<std::pair<int, int>>> symmetry_listH;
		x_splits = findSymmetryH(blurred_gray_img, w_range, SH_max, w_max, x_candidates, x_strong_splits, tau_max, cv::Range(0, blurred_gray_img.cols - 1), symmetry_listH);
		std::sort(x_splits.begin(), x_splits.end());
		
		if (x_splits.size() == 0 || x_splits[0] > 0) {
			x_splits.insert(x_splits.begin(), 0);
		}
		if (x_splits.back() < blurred_gray_img.cols - 1) {
			x_splits.push_back(blurred_gray_img.cols - 1);
		}
		

#if 0
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// DEBUG
		{
			cv::Mat_<float> SV_max;
			cv::Mat_<int> h_max;
			computeSV(img, SV_max, h_max, h_range);
			cv::Mat_<float> SH_max;
			cv::Mat_<int> w_max;
			computeSH(img, SH_max, w_max, w_range);

			outputFacadeStructure(img, y_splits, x_splits, std::string("../subdivision/") + filename, cv::Scalar(0, 255, 255), 1);
			outputFacadeStructure(img, SV_max, Ver, h_max, y_splits, SH_max, Hor, w_max, x_splits, std::string("../grad/") + filename, cv::Scalar(0, 255, 255), 1);
			//outputImageWithHorizontalAndVerticalGraph(orig_img, Ver, y_splits, Hor, x_splits, std::string("../grad/") + filename, 1);
		}
#endif

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// windows
#if 1
		{
			// compute gradient magnitude
			cv::Mat sobelx;
			cv::Sobel(gray_img, sobelx, CV_32F, 1, 0);
			cv::multiply(sobelx, sobelx, sobelx);

			cv::Mat sobely;
			cv::Sobel(gray_img, sobely, CV_32F, 0, 1);
			cv::multiply(sobely, sobely, sobely);

			cv::Mat_<float> grad_mag;
			cv::sqrt(sobelx + sobely, grad_mag);

			win_rects.resize(y_splits.size() - 1);
			for (int i = 0; i < y_splits.size() - 1; ++i) {
				win_rects[i].resize(x_splits.size() - 1);
				for (int j = 0; j < x_splits.size() - 1; ++j) {
					int x1 = x_splits[j];
					int x2 = x_splits[j + 1] - 1;
					int y1 = y_splits[i];
					int y2 = y_splits[i + 1] - 1;
					int w = x2 - x1 + 1;
					int h = y2 - y1 + 1;

					// for too small region, no window is created
					if (h < average_floor_height * 0.25 || w < average_floor_height * 0.2) {
						win_rects[i][j].valid = WindowPos::INVALID;
						continue;
					}
					
					// check the gradient magnitude along the boundary
					float total_mag = 0;
					for (int xx = x1; xx <= x2; ++xx) {
						total_mag += grad_mag(y1, xx);
						total_mag += grad_mag(y2, xx);
					}
					for (int yy = y1; yy <= y2; ++yy) {
						total_mag += grad_mag(yy, x1);
						total_mag += grad_mag(yy, x2);
					}

					// compute the average magnitude along the boundary
					float avg_mag = total_mag / (w * 2 + h * 2);

					// defind a threshold
					float threshold1 = avg_mag;
					float threshold2 = avg_mag * 3;

					// detect edge
					cv::Mat roi(gray_img, cv::Rect(x1, y1, w, h));
					cv::Mat edges;
					cv::Canny(roi, edges, threshold1, threshold2);

					// sum up edge horizontally and vertically
					cv::Mat edgeV, edgeH;
					cv::reduce(edges, edgeV, 1, cv::REDUCE_SUM, CV_32F);
					cv::reduce(edges, edgeH, 0, cv::REDUCE_SUM, CV_32F);
					edgeH = edgeH.t();

					// find the left edge of the window
					int left = -1;
					for (int xx = 0; xx < w; ++xx) {
						if (edgeH.at<float>(xx, 0) >= 255 * h * 0.2) {
							left = xx;
							break;
						}
					}

					// find the right edge of the window
					int right = -1;
					for (int xx = w - 1; xx >= 0; --xx) {
						if (edgeH.at<float>(xx, 0) >= 255 * h * 0.2) {
							right = xx;
							break;
						}
					}

					// find the top edge of the window
					int top = -1;
					for (int yy = 0; yy < h; ++yy) {
						if (edgeV.at<float>(yy, 0) >= 255 * w * 0.2) {
							top = yy;
							break;
						}
					}

					// find the bottom edge of the window
					int bottom = -1;
					for (int yy = h - 1; yy >= 0; --yy) {
						if (edgeV.at<float>(yy, 0) >= 255 * w * 0.2) {
							bottom = yy;
							break;
						}
					}

					if (left >= 0 && right >= 0 && left < right && top >= 0 && bottom >= 0 && top < bottom) {
						win_rects[i][j] = WindowPos(left, top, right, bottom);
					}
					else {
						win_rects[i][j].valid = WindowPos::INVALID;
					}
				}
			}

		}
#endif

	}

	std::vector<float> findSymmetryV(const cv::Mat& img, const cv::Range& h_range, std::map<int, float>& S_max, std::map<int, int>& h_max, const std::vector<int>& y_candidates, std::map<int, bool>& strong_splits, float tau_max, cv::Range range, std::vector<std::vector<std::pair<int, int>>>& symmetry_list) {
		std::vector<float> splits;

		if (range.end - range.start <= 1) return splits;

		// compute S_max and h_max at ys where Ver is local minimum
		int best_r = -1;
		float best_S;
		int best_h;
		for (int i = 0; i < y_candidates.size(); ++i) {
			int r = y_candidates[i];

			// too small region should not be considered for symmetry
			if (h_max[r] <= 2) continue;
			
			// skip a weak split at first try
			if (!strong_splits[r]) continue;

			if (r - h_max[r] < range.start || r + h_max[r] >= range.end) continue;

			best_r = r;
			best_S = S_max[r];
			best_h = h_max[r];
			break;
		}

		// second try
		if (best_r == -1) {
			for (int i = 0; i < y_candidates.size(); ++i) {
				int r = y_candidates[i];

				// too small region should not be considered for symmetry
				if (h_max[r] <= 2) continue;

				// skip a strong split in the second try
				if (strong_splits[r]) continue;

				if (r - h_max[r] < range.start || r + h_max[r] >= range.end) continue;

				best_r = r;
				best_S = S_max[r];
				best_h = h_max[r];
				break;
			}
		}

		// stop here if no max S is found.
		if (best_r == -1) return splits;

		// add new split
		splits.push_back(best_r);
		//std::cout << "new split line: " << best_r << std::endl;

		// add a symmetry
		if (best_S >= tau_max * 0.75) {
			symmetry_list.resize(symmetry_list.size() + 1);
			symmetry_list.back().push_back(std::make_pair(best_r - best_h, best_h));
			symmetry_list.back().push_back(std::make_pair(best_r, best_h));
		}

		// find the symmetry downward
		int cur = best_r;
		if (best_S >= tau_max * 0.75) {
			cur += best_h;
			while (cur <= range.end) {
				splits.push_back(cur);
				//std::cout << "    add split line: " << cur << std::endl;

				float SV;
				int h;
				computeSV(img, cur, SV, h, h_range);
				if (SV < tau_max * 0.75) break;
				if (abs(h - best_h) >  best_h * 0.1) break;

				symmetry_list.back().push_back(std::make_pair(cur, h));
				cur += h;
			}
		}

		// find the symmetry downward recursively
		std::vector<float> new_splits = findSymmetryV(img, h_range, S_max, h_max, y_candidates, strong_splits, tau_max, cv::Range(cur, range.end), symmetry_list);
		splits.insert(splits.end(), new_splits.begin(), new_splits.end());
		
		// find the symmetry upward
		cur = best_r;
		if (best_S >= tau_max * 0.75) {
			cur -= best_h;
			while (cur >= range.start) {				
				splits.push_back(cur);
				//std::cout << "    add split line: " << cur << std::endl;

				float SV;
				int h;
				computeSV(img, cur, SV, h, h_range);
				if (SV < tau_max * 0.75) break;
				if (abs(h - best_h) > best_h * 0.1) break;

				symmetry_list.back().push_back(std::make_pair(cur - h, h));
				cur -= h;
			}
		}

		// find the symmetry upward recursively
		new_splits = findSymmetryV(img, h_range, S_max, h_max, y_candidates, strong_splits, tau_max, cv::Range(range.start, cur), symmetry_list);
		splits.insert(splits.end(), new_splits.begin(), new_splits.end());
		
		return splits;
	}

	std::vector<float> findSymmetryH(const cv::Mat& img, const cv::Range& h_range, std::map<int, float>& S_max, std::map<int, int>& h_max, const std::vector<int>& x_candidates, std::map<int, bool>& strong_splits, float tau_max, cv::Range range, std::vector<std::vector<std::pair<int, int>>>& symmetry_list) {
		std::vector<float> splits;

		if (range.end - range.start <= 1) return splits;

		// compute S_max and h_max at ys where Ver is local minimum
		int best_r = -1;
		float best_S;
		int best_h;
		for (int i = 0; i < x_candidates.size(); ++i) {
			int r = x_candidates[i];

			// too small region should not be considered for symmetry
			if (h_max[r] <= 2) continue;

			// skip a weak split at first try
			if (!strong_splits[r]) continue;

			if (r - h_max[r] < range.start || r + h_max[r] > range.end) continue;

			best_r = r;
			best_S = S_max[r];
			best_h = h_max[r];
			break;
		}

		// second try
		if (best_r == -1) {
			for (int i = 0; i < x_candidates.size(); ++i) {
				int r = x_candidates[i];

				// too small region should not be considered for symmetry
				if (h_max[r] <= 2) continue;

				// skip a strong split in the second try
				if (strong_splits[r]) continue;

				if (r - h_max[r] < range.start || r + h_max[r] > range.end) continue;

				best_r = r;
				best_S = S_max[r];
				best_h = h_max[r];
				break;
			}
		}

		// stop here if no max S is found.
		if (best_r == -1) return splits;

		// add new split
		splits.push_back(best_r);
		//std::cout << "new split line: " << best_r << std::endl;

		// add a symmetry
		if (best_S >= tau_max * 0.75) {
			symmetry_list.resize(symmetry_list.size() + 1);
			symmetry_list.back().push_back(std::make_pair(best_r - best_h, best_h));
			symmetry_list.back().push_back(std::make_pair(best_r, best_h));
		}

		// find the symmetry downward
		int cur = best_r;
		if (best_S >= tau_max * 0.75) {
			cur += best_h;
			while (cur <= range.end) {
				splits.push_back(cur);
				//std::cout << "    add split line: " << cur << std::endl;

				float SH;
				int h;
				computeSH(img, cur, SH, h, h_range);
				if (SH < tau_max * 0.75) break;
				if (abs(h - best_h) >  best_h * 0.1) break;

				symmetry_list.back().push_back(std::make_pair(cur, h));
				cur += h;
			}
		}

		// find the symmetry downward recursively
		std::vector<float> new_splits = findSymmetryH(img, h_range, S_max, h_max, x_candidates, strong_splits, tau_max, cv::Range(cur, range.end), symmetry_list);
		splits.insert(splits.end(), new_splits.begin(), new_splits.end());
		
		// find the symmetry upward
		cur = best_r;
		if (best_S >= tau_max * 0.75) {
			cur -= best_h;
			while (cur >= range.start) {
				splits.push_back(cur);
				//std::cout << "    add split line: " << cur << std::endl;

				float SH;
				int h;
				computeSH(img, cur, SH, h, h_range);
				if (SH < tau_max * 0.75) break;
				if (abs(h - best_h) >  best_h * 0.1) break;

				symmetry_list.back().push_back(std::make_pair(cur - h, h));
				cur -= h;
			}
		}

		// find the symmetry downward recursively
		new_splits = findSymmetryH(img, h_range, S_max, h_max, x_candidates, strong_splits, tau_max, cv::Range(range.start, cur), symmetry_list);
		splits.insert(splits.end(), new_splits.begin(), new_splits.end());
		
		return splits;
	}

	/**
	* 2つの領域の類似度を返却する。
	*
	* @param R1		領域1 (1-channel image)
	* @param R2		領域2 (1-channel image)
	* @return			類似度
	*/
	float MI(const cv::Mat& R1, const cv::Mat& R2) {
#if 1
		cv::Mat_<float> Pab(256, 256, 0.0f);
		cv::Mat_<float> Pa(256, 1, 0.0f);
		cv::Mat_<float> Pb(256, 1, 0.0f);

		// create a histogram of intensities
		for (int r = 0; r < R1.rows; ++r) {
			for (int c = 0; c < R1.cols; ++c) {
				int a = R1.at<unsigned char>(r, c);
				int b = R2.at<unsigned char>(r, c);

				Pab(a, b)++;
				Pa(a, 0)++;
				Pb(b, 0)++;
			}
		}

		// normalize the historgram
		cv::Mat Pab_max;
		cv::Mat Pa_max;
		cv::Mat Pb_max;
		cv::reduce(Pab, Pab_max, 0, CV_REDUCE_SUM);
		cv::reduce(Pab_max, Pab_max, 1, CV_REDUCE_SUM);
		cv::reduce(Pa, Pa_max, 0, CV_REDUCE_SUM);
		cv::reduce(Pb, Pb_max, 0, CV_REDUCE_SUM);
		Pab /= Pab_max.at<float>(0, 0);
		Pa /= Pa_max.at<float>(0, 0);
		Pb /= Pb_max.at<float>(0, 0);

		float result = 0.0f;
		for (int a = 0; a < 256; ++a) {
			for (int b = 0; b < 256; ++b) {
				float v = Pab(a, b);
				if (v == 0) {
					v = 0.001f * 0.001f;
				}
				float v1 = Pa(a, 0);
				if (v1 == 0) v1 = 0.001f;
				float v2 = Pb(b, 0);
				if (v2 == 0) v2 = 0.001f;

				float hoge = Pab(a, b) * log(v / v1 / v2);
				result += Pab(a, b) * log(v / v1 / v2);
			}
		}

		return result;
#endif
#if 0
		cv::Mat norm_R1;
		cv::Mat norm_R2;

		return expf(-cvutils::msd(R1, R2) * 0.001f);
#endif
	}

	/**
	* Facade画像のS_max(y)、h_max(y)を計算する。
	*
	* @param img		Facade画像 (1-channel image)
	* @param SV_max		S_max(y)
	* @param h_max		h_max(y)
	* @param h_range	range of h
	*/
	void computeSV(const cv::Mat& img, cv::Mat_<float>& SV_max, cv::Mat_<int>& h_max, const cv::Range& h_range) {
		SV_max = cv::Mat_<float>(img.rows, 1, 0.0f);
		h_max = cv::Mat_<float>(img.rows, 1, 0.0f);

		printf("computing");
		for (int r = 0; r < img.rows; ++r) {
			printf("\rcomputing r = %d/%d  ", r, img.rows);

			computeSV(img, r, SV_max(r), h_max(r), h_range);
		}
		printf("\n");
	}

	void computeSV(const cv::Mat& img, int r, float& SV_max, int& h_max, const cv::Range& h_range) {
		SV_max = 0;
		h_max = 0;

		for (int h = h_range.start; h <= h_range.end; ++h) {
			if (r - h < 0 || r + h >= img.rows) continue;

			cv::Mat R1 = img(cv::Rect(0, r, img.cols, h));
			cv::Mat R2 = img(cv::Rect(0, r - h, img.cols, h));
			float SV = MI(R1, R2);

			if (SV > SV_max) {
				SV_max = SV;
				h_max = h;
			}
		}
	}

	/**
	* Facade画像のS_max(x)、w_max(x)を計算する。
	*
	* @param img		Facade画像 (1-channel image)
	* @param SH_max	S_max(x)
	* @param w_max		w_max(x)
	* @param w_range	range of w
	*/
	void computeSH(const cv::Mat& img, cv::Mat_<float>& SH_max, cv::Mat_<int>& w_max, const cv::Range& w_range) {
		SH_max = cv::Mat_<float>(img.cols, 1, 0.0f);
		w_max = cv::Mat_<float>(img.cols, 1, 0.0f);

		printf("computing");
		for (int c = 0; c < img.cols; ++c) {
			printf("\rcomputing c = %d/%d  ", c, img.cols);

			computeSH(img, c, SH_max(c), w_max(c), w_range);
		}
		printf("\n");
	}

	void computeSH(const cv::Mat& img, int c, float& SH_max, int& w_max, const cv::Range& w_range) {
		SH_max = 0;
		w_max = 0;

		for (int w = w_range.start; w <= w_range.end; ++w) {
			if (c - w < 0 || c + w >= img.cols) continue;

			cv::Mat R1 = img(cv::Rect(c, 0, w, img.rows));
			cv::Mat R2 = img(cv::Rect(c - w, 0, w, img.rows));
			float SH = MI(R1, R2);

			if (SH > SH_max) {
				SH_max = SH;
				w_max = w;
			}
		}
	}

	/**
	* imgから、Ver(y)とHor(x)を計算する。
	*
	* @param img		image (3-channel color image)
	* @param Ver		Ver(y)
	* @param Hor		Hor(x)
	*/
	void computeVerAndHor(const cv::Mat& img, cv::Mat_<float>& Ver, cv::Mat_<float>& Hor, float sigma) {
		cv::Mat grayImg;
		cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

		float alpha = 0.9f;

		// compute hor(x,y) and ver(x,y) according to Equation (3)
		cv::Mat_<float> hor(grayImg.rows, grayImg.cols);
		cv::Mat_<float> ver(grayImg.rows, grayImg.cols);
		for (int r = 0; r < grayImg.rows; ++r) {
			for (int c = 0; c < grayImg.cols; ++c) {
				float dIdx;
				float dIdy;

				if (c == 0) {
					dIdx = grayImg.at<unsigned char>(r, c + 1) - grayImg.at<unsigned char>(r, c);
				}
				else if (c == grayImg.cols - 1) {
					dIdx = grayImg.at<unsigned char>(r, c) - grayImg.at<unsigned char>(r, c - 1);
				}
				else {
					dIdx = (grayImg.at<unsigned char>(r, c + 1) - grayImg.at<unsigned char>(r, c - 1)) / 2.0f;
				}

				if (r == 0) {
					dIdy = grayImg.at<unsigned char>(r + 1, c) - grayImg.at<unsigned char>(r, c);
				}
				else if (r == grayImg.rows - 1) {
					dIdy = grayImg.at<unsigned char>(r, c) - grayImg.at<unsigned char>(r - 1, c);
				}
				else {
					dIdy = (float)(grayImg.at<unsigned char>(r + 1, c) - grayImg.at<unsigned char>(r - 1, c)) / 2.0f;
				}

				hor(r, c) = std::max(dIdy * dIdy * (1.0f - alpha) - dIdx * dIdx * alpha, 0.0f);
				ver(r, c) = std::max(dIdx * dIdx * (1.0f - alpha) - dIdy * dIdy * alpha, 0.0f);
			}
		}

		// sum up the ver(x, y) horizontally and vertically, respectively
		cv::Mat ver_xtotal;
		cv::Mat ver_ytotal;
		cv::reduce(ver, ver_xtotal, 1, cv::REDUCE_SUM);
		cv::reduce(ver, ver_ytotal, 0, cv::REDUCE_SUM);

		// sum up the hor(x, y) horizontally and vertically, respectively
		cv::Mat hor_xtotal;
		cv::Mat hor_ytotal;
		cv::reduce(hor, hor_xtotal, 1, cv::REDUCE_SUM);
		cv::reduce(hor, hor_ytotal, 0, cv::REDUCE_SUM);

		// compute Ver(y) and Hor(x) according to Equation (4)
		Ver = cv::Mat_<float>(grayImg.rows, 1, 0.0f);
		Hor = cv::Mat_<float>(1, grayImg.cols, 0.0f);
		float beta = 0.1f;
		for (int r = 0; r < grayImg.rows; ++r) {
			for (int rr = 0; rr < grayImg.rows; ++rr) {
				Ver(r, 0) += (ver_xtotal.at<float>(rr, 0) - beta * hor_xtotal.at<float>(rr, 0)) * utils::gause(rr - r, sigma);
			}
		}
		for (int c = 0; c < grayImg.cols; ++c) {
			for (int cc = 0; cc < grayImg.cols; ++cc) {
				Hor(0, c) += (hor_ytotal.at<float>(0, cc) - beta * ver_ytotal.at<float>(0, cc)) * utils::gause(cc - c, sigma);
			}
		}
	}

	/**
	 * 俺の方式。
	 */
	void computeVerAndHor2(const cv::Mat& img, cv::Mat_<float>& Ver, cv::Mat_<float>& Hor) {
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		// smoothing
		//cv::GaussianBlur(grayImg, grayImg, cv::Size(5, 5), 5, 5);

		// compute gradient magnitude
		cv::Mat sobelx;
		cv::Sobel(grayImg, sobelx, CV_32F, 1, 0);
		sobelx = cv::abs(sobelx);
		cv::Mat sobely;
		cv::Sobel(grayImg, sobely, CV_32F, 0, 1);
		sobely = cv::abs(sobely);

		// sum up the gradient magnitude horizontally and vertically
		cv::reduce(sobely, Hor, 0, CV_REDUCE_SUM);
		cv::reduce(sobelx, Ver, 1, CV_REDUCE_SUM);

		Hor = Hor.t();
	}

	/**
	 * tile内のwindowを検出し、その矩形座標を返却する。
	 * windowが検出されなかった場合はfalseを返却する。
	 *
	 * @param tile					タイル画像 (3-channel image)
	 * @param min_size
	 * @param horizontal_edge_max	水平分割線に対する、エッジの強さの最小値
	 * @param rect
	 * @return						分割する場合はtrue / false otherwise
	 */
	bool subdivideTile(const cv::Mat& tile, const cv::Mat& edges, int min_size, int tile_margin, WindowPos& winpos) {
		if (tile.cols < min_size || tile.rows < min_size) {
			winpos.valid = WindowPos::INVALID;
			return false;
		}

		// sum horizontally and vertically
		cv::Mat vertical_edges;
		cv::Mat horizontal_edges;
		cv::reduce(edges, vertical_edges, 0, CV_REDUCE_SUM, CV_32F);
		cv::reduce(edges, horizontal_edges, 1, CV_REDUCE_SUM, CV_32F);

		cv::Mat vertical_edges_max;
		cv::Mat horizonta_edges_max;
		cv::reduce(vertical_edges, vertical_edges_max, 1, CV_REDUCE_MAX, CV_32F);
		cv::reduce(horizontal_edges, horizonta_edges_max, 0, CV_REDUCE_MAX, CV_32F);

		float vertical_edge_threshold = tile.rows * 0.2f * 255;
		float horizontal_edge_threshold = tile.cols * 0.2f * 255;

		// find the  vertical edges (or x coordinates) that are closest to the side boundary
		int x1 = -1;
		float prev_x1;
		for (int c = tile_margin; c < vertical_edges.cols - tile_margin; ++c) {
			if (x1 == -1) {
				if (vertical_edges.at<float>(0, c) >= vertical_edge_threshold) {
					x1 = c;
					prev_x1 = vertical_edges.at<float>(0, c);
				}
			}
			else if (cvutils::isLocalMaximum(vertical_edges, c, 3)) {
				x1 = c;
				prev_x1 = vertical_edges.at<float>(0, c);
			}
			else {
				break;
			}

		}
		int x2 = -1;
		float prev_x2;
		for (int c = vertical_edges.cols - tile_margin - 1; c >= tile_margin; --c) {
			if (x2 == -1) {
				if (vertical_edges.at<float>(0, c) >= vertical_edge_threshold) {
					x2 = c;
					prev_x2 = vertical_edges.at<float>(0, c);
				}
			}
			else if (cvutils::isLocalMaximum(vertical_edges, c, 3)) {
				x2 = c;
				prev_x2 = vertical_edges.at<float>(0, c);
			}
			else {
				break;
			}
		}
		if (x1 == -1 || x2 == -1 || x2 - x1 <= 1) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		// find the horizontqal edges (or y coordinates) that are closest to the top and bottom boundaries
		int y1 = -1;
		float prev_y1;
		for (int r = tile_margin; r < horizontal_edges.rows - tile_margin; ++r) {
			if (y1 == -1) {
				if (horizontal_edges.at<float>(r, 0) >= horizontal_edge_threshold) {
					y1 = r;
					prev_y1 = horizontal_edges.at<float>(r, 0);
				}
			}
			else if (cvutils::isLocalMaximum(horizontal_edges, r, 3)) {
				y1 = r;
				prev_y1 = horizontal_edges.at<float>(r, 0);
			}
			else {
				break;
			}
		}
		int y2 = -1;
		float prev_y2;
		for (int r = horizontal_edges.rows - tile_margin - 1; r >= tile_margin; --r) {
			if (y2 == -1) {
				if (horizontal_edges.at<float>(r, 0) >= horizontal_edge_threshold) {
					y2 = r;
					prev_y2 = horizontal_edges.at<float>(r, 0);
				}
			}
			else if (cvutils::isLocalMaximum(horizontal_edges, r, 3)) {
				y2 = r;
				prev_y2 = horizontal_edges.at<float>(r, 0);
			}
			else {
				break;
			}
		}
		if (y1 == -1 || y2 == -1 || y2 - y1 <= 1) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		if ((float)(x2 - x1) / (y2 - y1) > 8.0f || (float)(y2 - y1) / (x2 - x1) > 8.0f) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		winpos = WindowPos(x1, y1, tile.cols - 1 - x2, tile.rows - 1 - y2);

		return true;
	}

	/**
	* tile内のwindowを検出し、その矩形座標を返却する。
	* windowが検出されなかった場合はfalseを返却する。
	*
	* @param tile					タイル画像 (3-channel image)
	* @param min_size
	* @param horizontal_edge_max	水平分割線に対する、エッジの強さの最小値
	* @param rect
	* @return						分割する場合はtrue / false otherwise
	*/
	bool subdivideTile2(const cv::Mat& tile, cv::Mat Ver, cv::Mat Hor, int min_size, int tile_margin, WindowPos& winpos) {
		if (tile.cols < min_size || tile.rows < min_size) {
			winpos.valid = WindowPos::INVALID;
			return false;
		}

		//cv::imwrite("tile.png", tile);

		outputImageWithHorizontalAndVerticalGraph(tile, Ver, Hor, "graph.png");

		double Ver_min, Ver_max;
		cv::minMaxLoc(Ver, &Ver_min, &Ver_max);
		double Hor_min, Hor_max;
		cv::minMaxLoc(Hor, &Hor_min, &Hor_max);

		float vertical_edge_threshold = (Ver_max - Ver_min) * 0.25f + Ver_min;
		float horizontal_edge_threshold = (Hor_max - Hor_max) * 0.25f + Ver_min;


		// find the  vertical edges (or x coordinates) that are closest to the side boundary
		int x1 = -1;
		float prev_x1;
		for (int c = tile_margin; c < Hor.cols - tile_margin; ++c) {
			if (x1 == -1) {
				if (Hor.at<float>(0, c) >= horizontal_edge_threshold) {
					x1 = c;
					prev_x1 = Hor.at<float>(0, c);
				}
			}
			else if (Hor.at<float>(0, c) > prev_x1) {
				x1 = c;
				prev_x1 = Hor.at<float>(0, c);
			}
			else {
				break;
			}

		}
		int x2 = -1;
		float prev_x2;
		for (int c = Hor.cols - tile_margin - 1; c >= tile_margin; --c) {
			if (x2 == -1) {
				if (Hor.at<float>(0, c) >= horizontal_edge_threshold) {
					x2 = c;
					prev_x2 = Hor.at<float>(0, c);
				}
			}
			else if (Hor.at<float>(0, c) > prev_x2) {
				x2 = c;
				prev_x2 = Hor.at<float>(0, c);
			}
			else {
				break;
			}
		}
		if (x1 == -1 || x2 == -1 || x2 - x1 <= 1) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		// find the horizontqal edges (or y coordinates) that are closest to the top and bottom boundaries
		int y1 = -1;
		float prev_y1;
		for (int r = tile_margin; r < Ver.rows - tile_margin; ++r) {
			if (y1 == -1) {
				if (Ver.at<float>(r, 0) >= vertical_edge_threshold) {
					y1 = r;
					prev_y1 = Ver.at<float>(r, 0);
				}
			}
			else if (Ver.at<float>(r, 0) > prev_y1) {
				y1 = r;
				prev_y1 = Ver.at<float>(r, 0);
			}
			else {
				break;
			}
		}
		int y2 = -1;
		float prev_y2;
		for (int r = Ver.rows - tile_margin - 1; r >= tile_margin; --r) {
			if (y2 == -1) {
				if (Ver.at<float>(r, 0) >= vertical_edge_threshold) {
					y2 = r;
					prev_y2 = Ver.at<float>(r, 0);
				}
			}
			else if (Ver.at<float>(r, 0) > prev_y2) {
				y2 = r;
				prev_y2 = Ver.at<float>(r, 0);
			}
			else {
				break;
			}
		}
		if (y1 == -1 || y2 == -1 || y2 - y1 <= 1) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		if ((float)(x2 - x1) / (y2 - y1) > 8.0f || (float)(y2 - y1) / (x2 - x1) > 8.0f) {
			winpos.valid = WindowPos::UNCERTAIN;
			return false;
		}

		winpos = WindowPos(x1, y1, tile.cols - 1 - x2, tile.rows - 1 - y2);

		return true;
	}

	/**
	* Ver(y)の極小値をsplit lineの候補とし、S_max(y)に基づいて最適なsplit lineの組み合わせを探す。
	*
	* @param Ver			Ver(y)
	* @param min_interval	minimum floor height
	* @param max_interval	maximum floor height
	* @param y_split		最適なsplit lineの組み合わせ
	*/
	void findBestHorizontalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Ver, float min_interval, float max_interval, std::vector<int>& y_split) {
		y_split.clear();

		std::vector<int> y_candidates = cvutils::getPeak(Ver, false, 1, cvutils::LOCAL_MINIMUM, 1);
		y_candidates.insert(y_candidates.begin(), 0);
		y_candidates.push_back(img.rows - 1);

		std::vector<std::vector<float>> costs;
		std::vector<std::vector<int>> indices;
		std::vector<std::vector<int>> nums;

		// 最初の行のコストを初期化
		{
			costs.push_back(std::vector<float>(y_candidates.size(), std::numeric_limits<float>::max()));
			indices.push_back(std::vector<int>(y_candidates.size(), -1));
			nums.push_back(std::vector<int>(y_candidates.size(), 0));
			costs[0][0] = 0;
			indices[0][0] = 0;
			nums[0][0] = 0;
		}

		// 2行目以降について、Dynamic Programmingで最小コストを計算
		for (int i = 1;; ++i) {
			costs.push_back(std::vector<float>(y_candidates.size(), std::numeric_limits<float>::max()));
			indices.push_back(std::vector<int>(y_candidates.size(), -1));
			nums.push_back(std::vector<int>(y_candidates.size(), 0));

			for (int k = 0; k < y_candidates.size() - 1; ++k) {
				bool found = false;

				for (int j = k + 1; j < y_candidates.size() - 1; ++j) {
					if (indices[i - 1][k] == -1) continue;

					if (y_candidates[j] - y_candidates[k] < min_interval) continue;
					if (found && y_candidates[j] - y_candidates[k] > max_interval) continue;

					found = true;
					float new_cost = costs[i - 1][k] + Ver(y_candidates[j], 0);
					if (new_cost / (nums[i - 1][k] + 1) < costs[i][j] / nums[i][j]) {
						costs[i][j] = new_cost;
						indices[i][j] = k;
						nums[i][j] = nums[i - 1][k] + 1;
					}
				}
			}

			for (int k = 0; k < y_candidates.size(); ++k) {
				if (indices[i - 1][k] == -1) continue;

				if (y_candidates.back() - y_candidates[k] > max_interval) continue;

				if (costs[i - 1][k] / nums[i - 1][k] < costs[i].back() / nums[i].back()) {
					costs[i].back() = costs[i - 1][k];
					indices[i].back() = k;
					nums[i].back() = nums[i - 1][k];
				}
			}

			// 最後のy以外がすべて-1なら、終了
			bool updated = false;
			for (int j = 0; j < indices[i].size() - 1; ++j) {
				if (indices[i][j] != -1) updated = true;
			}
			if (!updated) break;
		}

		// y_splitに、最適解を格納する
		y_split.push_back(y_candidates.back());
		int prev_index = y_candidates.size() - 1;
		for (int i = indices.size() - 1; i >= 1; --i) {
			int index = indices[i][prev_index];
			if (index == prev_index) continue;

			y_split.insert(y_split.begin(), y_candidates[index]);
			prev_index = index;
		}
	}

	/**
	* Hor(x)の極小値をsplit lineの候補とし、S_max(x)に基づいて最適なsplit lineの組み合わせを探す。
	*
	* @param Hor			Hor(x)
	* @param min_interval	minimum tile width
	* @param max_interval	maximum tile width
	* @param x_split		最適なsplit lineの組み合わせ
	*/
	void findBestVerticalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Hor, float min_interval, float max_interval, std::vector<int>& x_split) {
		x_split.clear();

		std::vector<int> x_candidates = cvutils::getPeak(Hor, false, 1, cvutils::LOCAL_MINIMUM, 1);
		x_candidates.insert(x_candidates.begin(), 0);
		x_candidates.push_back(img.cols - 1);

		std::vector<std::vector<float>> costs;
		std::vector<std::vector<int>> indices;
		std::vector<std::vector<int>> nums;

		// 最初の列のコストを初期化
		{
			costs.push_back(std::vector<float>(x_candidates.size(), std::numeric_limits<float>::max()));
			indices.push_back(std::vector<int>(x_candidates.size(), -1));
			nums.push_back(std::vector<int>(x_candidates.size(), 0));
			costs[0][0] = 0;
			indices[0][0] = 0;
			nums[0][0] = 0;
		}

		// 2列目以降について、Dynamic Programmingで最小コストを計算
		for (int i = 1;; ++i) {
			costs.push_back(std::vector<float>(x_candidates.size(), std::numeric_limits<float>::max()));
			indices.push_back(std::vector<int>(x_candidates.size(), -1));
			nums.push_back(std::vector<int>(x_candidates.size(), 0));

			for (int k = 0; k < x_candidates.size() - 1; ++k) {
				bool found = false;

				for (int j = k + 1; j < x_candidates.size() - 1; ++j) {
					if (indices[i - 1][k] == -1) continue;

					if (x_candidates[j] - x_candidates[k] < min_interval) continue;
					if (found && x_candidates[j] - x_candidates[k] > max_interval) continue;

					found = true;
					float new_cost = costs[i - 1][k] + Hor(0, x_candidates[j]);
					if (new_cost / (nums[i - 1][k] + 1) < costs[i][j] / nums[i][j]) {
						costs[i][j] = new_cost;
						indices[i][j] = k;
						nums[i][j] = nums[i - 1][k] + 1;
					}
				}
			}

			for (int k = 0; k < x_candidates.size(); ++k) {
				if (indices[i - 1][k] == -1) continue;

				if (x_candidates.back() - x_candidates[k] > max_interval) continue;

				if (costs[i - 1][k] / nums[i - 1][k] < costs[i].back() / nums[i].back()) {
					costs[i].back() = costs[i - 1][k];
					indices[i].back() = k;
					nums[i].back() = nums[i - 1][k];
				}
			}

			// 最後のx以外がすべて-1なら、終了
			bool updated = false;
			for (int j = 0; j < indices[i].size() - 1; ++j) {
				if (indices[i][j] != -1) updated = true;
			}
			if (!updated) break;
		}

		// x_splitに、最適解を格納する
		x_split.push_back(x_candidates.back());
		int prev_index = x_candidates.size() - 1;
		for (int i = indices.size() - 1; i >= 1; --i) {
			int index = indices[i][prev_index];
			if (index == prev_index) continue;

			x_split.insert(x_split.begin(), x_candidates[index]);
			prev_index = index;
		}
	}

	/**
	* 与えられた関数の極小値を使ってsplit lineを決定する。
	*/
	void getSplitLines(const cv::Mat_<float>& val, float threshold, std::vector<float>& split_positions) {
		cv::Mat_<float> mat = val.clone();
		if (mat.rows == 1) {
			mat = mat.t();
		}

		double max_value, min_value;
		cv::minMaxLoc(mat, &min_value, &max_value);
		threshold *= (max_value - min_value);

		for (int r = 0; r < mat.rows; ++r) {
			if (isLocalMinimum(mat, r, threshold)) {
				split_positions.push_back(r);
			}
		}

		// remove the both end
		if (split_positions.size() > 0) {
			if (split_positions[0] == 0) {
				split_positions.erase(split_positions.begin());
			}
			if (split_positions.back() == mat.rows - 1) {
				split_positions.pop_back();
			}
		}

		/*
		// 両端処理
		if (split_positions.size() == 0) {
			split_positions.insert(split_positions.begin(), 0);
		}
		else if (split_positions[0] > 0) {
			if (split_positions[0] < 5) {
				split_positions[0] = 0;
			}
			else {
				split_positions.insert(split_positions.begin(), 0);
			}
		}

		if (split_positions.back() < mat.rows - 1) {
			if (split_positions.back() >= mat.rows - 5) {
				split_positions.back() = mat.rows - 1;
			}
			else {
				split_positions.push_back(mat.rows - 1);
			}
		}
		*/
	}

	// 間隔が狭すぎる場合は、分割して隣接領域にマージする
	void refineSplitLines(std::vector<float>& split_positions, float threshold) {
		// 最大の間隔を計算する（ただし、１階は除く）
		float max_interval = 0;
		for (int i = 0; i < split_positions.size() - 2; ++i) {
			float interval = split_positions[i + 1] - split_positions[i];
			if (interval > max_interval) max_interval = interval;
		}

		while (true) {
			bool updated = false;
			for (int i = 0; i < split_positions.size() - 1;) {
				if (split_positions[i + 1] - split_positions[i] < max_interval * threshold) {
					if (i == 0) {
						split_positions.erase(split_positions.begin() + 1);
					}
					else if (i == split_positions.size() - 2) {
						split_positions.erase(split_positions.begin() + split_positions.size() - 2);
					}
					else {
						split_positions[i] = (split_positions[i + 1] + split_positions[i]) * 0.5;
						split_positions.erase(split_positions.begin() + i + 1);
					}
					updated = true;
				}
				else {
					i++;
				}
			}

			if (!updated) break;
		}
	}

	/**
	 * 分割線の一部を削除し、分割線の間隔が等間隔に近くなるようにする。
	 */
	void distributeSplitLines(std::vector<float>& split_positions, float threshold) {
		std::vector<int> flags(split_positions.size() - 2, 0);

		float min_stddev = std::numeric_limits<float>::max();
		std::vector<int> min_flags;

		while (true) {
			// count 1s
			int cnt_ones = 0;
			for (int i = 0; i < flags.size(); ++i) {
				if (flags[i] == 1) cnt_ones++;
			}

			// valid only if 1s are more than 50%
			if ((float)(cnt_ones + 2) / split_positions.size() > threshold) {
				// compute the distances between split lines
				std::vector<float> intervals;
				float prev_pos = split_positions[0];
				for (int i = 1; i < split_positions.size(); ++i) {
					if (i < split_positions.size() - 1 && flags[i - 1] == 0) continue;

					intervals.push_back(split_positions[i] - prev_pos);
					prev_pos = split_positions[i];
				}

				// compute the stddev of intervals
				float stddev = utils::stddev(intervals);

				// update the minimum stddev
				if (stddev < min_stddev) {
					min_stddev = stddev;
					min_flags = flags;
				}
			}

			// next permutation
			bool carried = false;
			for (int i = 0; i < flags.size(); ++i) {
				if (flags[i] == 1) {
					flags[i] = 0;
				}
				else if (flags[i] == 0) {
					flags[i] = 1;
					carried = true;
					break;
				}
			}
			if (!carried) break;
		}

		std::vector<float> tmp = split_positions;
		split_positions.clear();
		for (int i = 0; i < tmp.size(); ++i) {
			if (i == 0 || i == tmp.size() - 1) {
				split_positions.push_back(tmp[i]);
			}
			else if (min_flags[i - 1] == 1) {
				split_positions.push_back(tmp[i]);
			}
		}
	}

	void refine(std::vector<float>& y_split, std::vector<float>& x_split, std::vector<std::vector<WindowPos>>& winpos, float threshold) {
		// 各フロアの窓の数をカウントする
		std::vector<int> win_per_row(y_split.size() - 1, 0);
		int max_win_per_row = 0;
		for (int i = 0; i < y_split.size() - 1; ++i) {
			for (int j = 0; j < x_split.size() - 1; ++j) {
				if (winpos[i][j].valid == WindowPos::VALID) {
					win_per_row[i]++;
				}
			}
			if (win_per_row[i] > max_win_per_row) {
				max_win_per_row = win_per_row[i];
			}
		}

		// 各カラムの窓の数をカウントする
		std::vector<int> win_per_col(x_split.size() - 1, 0);
		int max_win_per_col = 0;
		for (int j = 0; j < x_split.size() - 1; ++j) {
			for (int i = 0; i < y_split.size() - 1; ++i) {
				if (winpos[i][j].valid == WindowPos::VALID) {
					win_per_col[j]++;
				}
			}
			if (win_per_col[j] > max_win_per_col) {
				max_win_per_col = win_per_col[j];
			}
		}

		// 壁のフロアかどうかチェックする
		std::vector<bool> is_wall_row(y_split.size() - 1, false);
		for (int i = 0; i < y_split.size() - 1; ++i) {
			if (win_per_row[i] < max_win_per_row * threshold) {
				is_wall_row[i] = true;

				for (int j = 0; j < x_split.size() - 1; ++j) {
					winpos[i][j].valid = WindowPos::INVALID;
				}
			}
		}

		// 壁のカラムかどうかチェックする
		std::vector<bool> is_wall_col(x_split.size() - 1, false);
		for (int j = 0; j < x_split.size() - 1; ++j) {
			if (win_per_col[j] < max_win_per_col * threshold) {
				is_wall_col[j] = true;

				for (int i = 0; i < y_split.size() - 1; ++i) {
					winpos[i][j].valid = WindowPos::INVALID;
				}
			}
		}

		// 窓のないフロアが連続している場合は、連結する
		for (int i = 0; i < is_wall_row.size() - 1;) {
			if (is_wall_row[i] && is_wall_row[i + 1]) {
				is_wall_row.erase(is_wall_row.begin() + i + 1);
				y_split.erase(y_split.begin() + i + 1);
				winpos.erase(winpos.begin() + i + 1);
			}
			else {
				i++;
			}
		}

		// 窓のないカラムが連続している場合は、連結する
		for (int j = 0; j < is_wall_col.size() - 1;) {
			if (is_wall_col[j] && is_wall_col[j + 1]) {
				is_wall_col.erase(is_wall_col.begin() + j + 1);
				x_split.erase(x_split.begin() + j + 1);
				for (int i = 0; i < y_split.size() - 1; ++i) {
					winpos[i].erase(winpos[i].begin() + j + 1);
				}
			}
			else {
				j++;
			}
		}
	}

	void align(const cv::Mat& edge_img, const std::vector<float>& y_split, const std::vector<float>& x_split, std::vector<std::vector<WindowPos>> &winpos, int max_iter) {
		// 窓のX座標をvoteする
		for (int j = 0; j < x_split.size() - 1; ++j) {
			int max_left, max_right;

			// voteする
			std::vector<float> histogram1(x_split[j + 1] - x_split[j], 0);
			std::vector<float> histogram2(x_split[j + 1] - x_split[j], 0);
			int count = 0;
			for (int i = 0; i < y_split.size() - 1; ++i) {
				if (!winpos[i][j].valid) continue;

				count++;
				for (int c = 0; c < histogram1.size(); ++c) {
					histogram1[c] += utils::gause(winpos[i][j].left - c, 2);
					histogram2[c] += utils::gause(winpos[i][j].right - c, 2);
				}
			}

			if (count == 0) continue;

			// max voteを探す
			float max_val1 = 0.0f;
			float max_val2 = 0.0f;
			for (int c = 0; c < histogram1.size(); ++c) {
				if (histogram1[c] > max_val1) {
					max_val1 = histogram1[c];
					max_left = c;
				}
				if (histogram2[c] > max_val2) {
					max_val2 = histogram2[c];
					max_right = c;
				}
			}

			// 全てのフロアの窓のX座標をそろえる
			for (int r = 0; r < y_split.size() - 1; ++r) {
				if (!winpos[r][j].valid) continue;

				if (r == 0 || r == y_split.size() - 1) {
					if (abs(winpos[r][j].left - max_left) < 5) {
						winpos[r][j].left = max_left;
					}
					if (abs(winpos[r][j].right - max_right) < 5) {
						winpos[r][j].right = max_right;
					}
				}
				else {
					winpos[r][j].left = max_left;
					winpos[r][j].right = max_right;
				}
			}
		}

		// 窓のY座標をvoteする
		for (int i = 0; i < y_split.size() - 1; ++i) {
			int max_top, max_bottom;

			// voteする
			std::vector<float> histogram1(y_split[i + 1] - y_split[i], 0);
			std::vector<float> histogram2(y_split[i + 1] - y_split[i], 0);
			int count = 0;
			for (int j = 0; j < x_split.size() - 1; ++j) {
				if (!winpos[i][j].valid) continue;

				count++;
				for (int r = 0; r < histogram1.size(); ++r) {
					histogram1[r] += utils::gause(winpos[i][j].top - r, 2);
					histogram2[r] += utils::gause(winpos[i][j].bottom - r, 2);
				}
			}

			if (count == 0) continue;

			// max voteを探す
			float max_val1 = 0.0f;
			float max_val2 = 0.0f;
			for (int r = 0; r < histogram1.size(); ++r) {
				if (histogram1[r] > max_val1) {
					max_val1 = histogram1[r];
					max_top = r;
				}
				if (histogram2[r] > max_val2) {
					max_val2 = histogram2[r];
					max_bottom = r;
				}
			}

			// 全てのカラムの窓のY座標をそろえる
			for (int c = 0; c < x_split.size() - 1; ++c) {
				if (!winpos[i][c].valid) continue;

				winpos[i][c].top = max_top;
				winpos[i][c].bottom = max_bottom;
			}
		}
	}

	bool isLocalMinimum(const cv::Mat& mat, int index, float threshold) {
		float origin_value = mat.at<float>(index);

		// check upward
		for (int r = index - 1; r >= 0; --r) {
			if (mat.at<float>(r) < origin_value) return false;
			if (fabs(mat.at<float>(r) - origin_value) > threshold) break;
		}

		// check downward
		for (int r = index + 1; r < mat.rows; ++r) {
			if (mat.at<float>(r) < origin_value) return false;
			if (fabs(mat.at<float>(r) - origin_value) > threshold) break;
		}

		return true;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// visualization

	void outputFacadeStructure(cv::Mat img, const std::vector<float>& y_splits, const std::vector<float>& x_splits, const std::string& filename, cv::Scalar lineColor, int lineWidth) {
		cv::Mat result;
		if (img.channels() == 1) {
			cv::cvtColor(img, result, cv::COLOR_GRAY2BGR);
		}
		else {
			result = img.clone();
		}

		for (int i = 0; i < y_splits.size(); ++i) {
			if (i < y_splits.size() - 1) {
				cv::line(result, cv::Point(0, y_splits[i]), cv::Point(img.cols, y_splits[i]), lineColor, lineWidth);
			}
			else {
				// For the last line, we need to move the line upward by 1px to make it inside the image.
				cv::line(result, cv::Point(0, y_splits[i] - 1), cv::Point(img.cols, y_splits[i] - 1), lineColor, lineWidth);
			}
		}
		for (int i = 0; i < x_splits.size(); ++i) {
			if (i < x_splits.size() - 1) {
				cv::line(result, cv::Point(x_splits[i], 0), cv::Point(x_splits[i], img.rows), lineColor, lineWidth);
			}
			else {
				// For the last line, we need to move the line upward by 1px to make it inside the image.
				cv::line(result, cv::Point(x_splits[i] - 1, 0), cv::Point(x_splits[i] - 1, img.rows), lineColor, lineWidth);
			}
		}
		cv::imwrite(filename, result);
	}

	void outputFacadeStructure(cv::Mat img, const cv::Mat_<float>& SV_max, const cv::Mat_<float>& Ver, const cv::Mat_<float>& h_max, const std::vector<float>& y_splits, const cv::Mat_<float>& SH_max, const cv::Mat_<float>& Hor, const cv::Mat_<float>& w_max, const std::vector<float>& x_splits, const std::string& filename, cv::Scalar lineColor, int lineWidth) {
		if (img.channels() == 1) {
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		}

		double min_SV, max_SV;
		double min_Ver, max_Ver;
		double min_h, max_h;
		double min_SH, max_SH;
		double min_Hor, max_Hor;
		double min_w, max_w;
		cv::minMaxLoc(SV_max, &min_SV, &max_SV);
		cv::minMaxLoc(Ver, &min_Ver, &max_Ver);
		cv::minMaxLoc(h_max, &min_h, &max_h);
		cv::minMaxLoc(SH_max, &min_SH, &max_SH);
		cv::minMaxLoc(Hor, &min_Hor, &max_Hor);
		cv::minMaxLoc(h_max, &min_w, &max_w);

		int graphSizeV = img.rows * 0.25;
		int marginV = graphSizeV * 0.2;
		int graphSizeH = std::max(80.0, img.rows * 0.25);
		int marginH = graphSizeH * 0.2;
		cv::Mat result(img.rows + graphSizeH * 2 + max_w + marginH * 4, img.cols + graphSizeV * 2 + max_h + marginV * 4, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw SV_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + marginV + (SV_max(r) - min_SV) / (max_SV - min_SV) * graphSizeV;
			int x2 = img.cols + marginV + (SV_max(r + 1) - min_SV) / (max_SV - min_SV) * graphSizeV;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Ver
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSizeV + marginV * 2 + (Ver(r) - min_Ver) / (max_Ver - min_Ver) * graphSizeV;
			int x2 = img.cols + graphSizeV + marginV * 2 + (Ver(r + 1) - min_Ver) / (max_Ver - min_Ver) * graphSizeV;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw h_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSizeV * 2 + marginV * 3 + h_max(r);
			int x2 = img.cols + graphSizeV * 2 + marginV * 3 + h_max(r + 1);

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw y splits
		for (int i = 0; i < y_splits.size(); ++i) {
			cv::line(result, cv::Point(0, y_splits[i]), cv::Point(img.cols, y_splits[i]), cv::Scalar(0, 255, 255), 3);
		}

		// draw SH_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + marginH + (SH_max(c) - min_SH) / (max_SH - min_SH) * graphSizeH;
			int y2 = img.rows + marginH + (SH_max(c + 1) - min_SH) / (max_SH - min_SH) * graphSizeH;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Hor
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSizeH + marginH * 2 + (Hor(c) - min_Hor) / (max_Hor - min_Hor) * graphSizeH;
			int y2 = img.rows + graphSizeH + marginH * 2 + (Hor(c + 1) - min_Hor) / (max_Hor - min_Hor) * graphSizeH;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw w_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSizeH * 2 + marginH * 3 + w_max(c);
			int y2 = img.rows + graphSizeH * 2 + marginH * 3 + w_max(c + 1);

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw x splits
		for (int i = 0; i < x_splits.size(); ++i) {
			cv::line(result, cv::Point(x_splits[i], 0), cv::Point(x_splits[i], img.rows), cv::Scalar(0, 255, 255), 3);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeAndWindows(const cv::Mat& img, const std::vector<float>& y_split, const std::vector<float>& x_split, const std::vector<std::vector<WindowPos>>& winpos, const std::string& filename, cv::Scalar lineColor, int lineWidth) {
		cv::Mat result = img.clone();
		for (int i = 0; i < y_split.size(); ++i) {
			if (i < y_split.size() - 1) {
				cv::line(result, cv::Point(0, y_split[i]), cv::Point(result.cols - 1, y_split[i]), cv::Scalar(0, 0, 255), lineWidth);
			}
			else {
				cv::line(result, cv::Point(0, y_split[i] - 1), cv::Point(result.cols - 1, y_split[i] - 1), cv::Scalar(0, 0, 255), lineWidth);
			}
		}
		for (int i = 0; i < x_split.size(); ++i) {
			if (i < x_split.size() - 1) {
				cv::line(result, cv::Point(x_split[i], 0), cv::Point(x_split[i], result.rows - 1), cv::Scalar(0, 0, 255), lineWidth);
			}
			else {
				cv::line(result, cv::Point(x_split[i] - 1, 0), cv::Point(x_split[i] - 1, result.rows - 1), cv::Scalar(0, 0, 255), lineWidth);
			}
		}
		for (int i = 0; i < y_split.size() - 1; ++i) {
			for (int j = 0; j < x_split.size() - 1; ++j) {
				if (winpos[i][j].valid == WindowPos::VALID) {
					int x = x_split[j] + winpos[i][j].left;
					int y = y_split[i] + winpos[i][j].top;
					int w = winpos[i][j].right - winpos[i][j].left + 1;
					int h = winpos[i][j].bottom - winpos[i][j].top + 1;
					cv::rectangle(result, cv::Rect(x, y, w, h), lineColor, lineWidth);
				}
			}
		}
		cv::imwrite(filename, result);
	}

	void outputWindows(const std::vector<float>& y_split, const std::vector<float>& x_split, const std::vector<std::vector<WindowPos>>& winpos, const std::string& filename, cv::Scalar lineColor, int lineWidth) {
		cv::Mat result(y_split.back(), x_split.back(), CV_8UC3, cv::Scalar(255, 255, 255));
		for (int i = 0; i < y_split.size() - 1; ++i) {
			for (int j = 0; j < x_split.size() - 1; ++j) {
				if (winpos[i][j].valid == WindowPos::VALID) {
					int x1 = x_split[j] + winpos[i][j].left;
					int y1 = y_split[i] + winpos[i][j].top;
					int x2 = x_split[j + 1] - 1 - winpos[i][j].right;
					int y2 = y_split[i + 1] - 1 - winpos[i][j].bottom;
					cv::rectangle(result, cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1), lineColor, lineWidth);
				}
			}
		}
		cv::imwrite(filename, result);
	}

	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const std::vector<float>& ys, const cv::Mat& hor, const std::vector<float>& xs, const std::string& filename, int lineWidth) {
		int graphSize = std::max(10.0, std::max(img.rows, img.cols) * 0.3);

		cv::Mat result;
		cv::Scalar graph_color;
		cv::Scalar peak_color;

		result = cv::Mat(img.rows + graphSize + 3, img.cols + graphSize + 3, CV_8UC3, cv::Scalar(255, 255, 255));
		graph_color = cv::Scalar(0, 0, 0);
		peak_color = cv::Scalar(0, 255, 255);

		// copy img to result
		cv::Mat color_img;
		if (img.channels() == 1) {
			cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);
		}
		else if (img.channels() == 3) {
			color_img = img;
		}
		cv::Mat roi(result, cv::Rect(0, 0, color_img.cols, color_img.rows));
		color_img.copyTo(roi);

		// get the maximum value of Ver(y) and Hor(x)
		float max_ver = cvutils::max(ver);
		float min_ver = cvutils::min(ver);
		float max_hor = cvutils::max(hor);
		float min_hor = cvutils::min(hor);

		// draw vertical graph
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + (cvutils::get(ver, r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + (cvutils::get(ver, r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), graph_color, 1, cv::LINE_8);
		}
		for (int i = 0; i < ys.size(); ++i) {
			cv::line(result, cv::Point(0, ys[i]), cv::Point(img.cols - 1, ys[i]), peak_color, lineWidth);
		}

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + (cvutils::get(hor, 0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + (cvutils::get(hor, 0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), graph_color, 1, cv::LINE_8);
		}
		for (int i = 0; i < xs.size(); ++i) {
			cv::line(result, cv::Point(xs[i], 0), cv::Point(xs[i], img.rows - 1), peak_color, lineWidth);
		}

		cv::imwrite(filename, result);
	}

	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const std::string& filename) {
		int graphSize = std::max(10.0, std::max(img.rows, img.cols) * 0.3);

		cv::Mat result;
		cv::Scalar graph_color;
		cv::Scalar peak_color;

		result = cv::Mat(img.rows + graphSize + 3, img.cols + graphSize + 3, CV_8UC3, cv::Scalar(255, 255, 255));
		graph_color = cv::Scalar(0, 0, 0);
		peak_color = cv::Scalar(0, 0, 255);

		// copy img to result
		cv::Mat color_img;
		if (img.channels() == 1) {
			cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);
		}
		else if (img.channels() == 3) {
			color_img = img;
		}
		cv::Mat roi(result, cv::Rect(0, 0, color_img.cols, color_img.rows));
		color_img.copyTo(roi);

		// get the maximum value of Ver(y) and Hor(x)
		float max_ver = cvutils::max(ver);
		float min_ver = cvutils::min(ver);
		float max_hor = cvutils::max(hor);
		float min_hor = cvutils::min(hor);

		// draw vertical graph
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + (cvutils::get(ver, r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + (cvutils::get(ver, r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), graph_color, 1, cv::LINE_8);
		}

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + (cvutils::get(hor, c, 0) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + (cvutils::get(hor, c + 1, 0) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), graph_color, 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	/**
	* Facade画像と合わせて、S_max(y)とh_max(y)を画像として保存する。
	* 論文 Fig 5に相当する画像。
	*
	* @param img		Facade画像
	* @param S_max		S_max
	* @param h_max		h_max
	* @param filename	output file name
	*/
	void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const std::string& filename) {
		float max_S = cvutils::max(S_max);
		float min_S = cvutils::min(S_max);
		float max_h = cvutils::max(h_max);
		float min_h = cvutils::min(h_max);

		int graphSize = img.rows * 0.25;
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows, img.cols + graphSize + max_h + margin * 3, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + margin + (S_max(r) - min_S) / (max_S - min_S) * graphSize;
			int x2 = img.cols + margin + (S_max(r + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw h_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSize + margin * 2 + h_max(r);
			int x2 = img.cols + graphSize + margin * 2 + h_max(r + 1);

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& Ver, const cv::Mat_<float>& h_max, const std::string& filename) {
		double min_S, max_S;
		double min_Ver, max_Ver;
		double min_h, max_h;
		cv::minMaxLoc(S_max, &min_S, &max_S);
		cv::minMaxLoc(Ver, &min_Ver, &max_Ver);
		cv::minMaxLoc(h_max, &min_h, &max_h);
		
		int graphSize = img.rows * 0.25;
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows, img.cols + graphSize * 2 + max_h + margin * 4, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + margin + (S_max(r) - min_S) / (max_S - min_S) * graphSize;
			int x2 = img.cols + margin + (S_max(r + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Ver
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSize + margin * 2 + (Ver(r) - min_Ver) / (max_Ver - min_Ver) * graphSize;
			int x2 = img.cols + graphSize + margin * 2 + (Ver(r + 1) - min_Ver) / (max_Ver - min_Ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw h_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSize * 2 + margin * 3 + h_max(r);
			int x2 = img.cols + graphSize * 2 + margin * 3 + h_max(r + 1);

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& Ver, const cv::Mat_<float>& h_max, const std::vector<float>& y_splits, const std::string& filename) {
		double min_S, max_S;
		double min_Ver, max_Ver;
		double min_h, max_h;
		cv::minMaxLoc(S_max, &min_S, &max_S);
		cv::minMaxLoc(Ver, &min_Ver, &max_Ver);
		cv::minMaxLoc(h_max, &min_h, &max_h);

		int graphSize = img.rows * 0.25;
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows, img.cols + graphSize * 2 + max_h + margin * 4, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + margin + (S_max(r) - min_S) / (max_S - min_S) * graphSize;
			int x2 = img.cols + margin + (S_max(r + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Ver
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSize + margin * 2 + (Ver(r) - min_Ver) / (max_Ver - min_Ver) * graphSize;
			int x2 = img.cols + graphSize + margin * 2 + (Ver(r + 1) - min_Ver) / (max_Ver - min_Ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw h_max
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + graphSize * 2 + margin * 3 + h_max(r);
			int x2 = img.cols + graphSize * 2 + margin * 3 + h_max(r + 1);

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw y splits
		for (int i = 0; i < y_splits.size(); ++i) {
			cv::line(result, cv::Point(0, y_splits[i]), cv::Point(img.cols, y_splits[i]), cv::Scalar(0, 255, 255), 3);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const std::string& filename) {
		float max_S = cvutils::max(S_max);
		float min_S = cvutils::min(S_max);
		float max_w = cvutils::max(w_max);
		float min_w = cvutils::min(w_max);

		int graphSize = std::max(80.0, img.rows * 0.25);
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows + graphSize + max_w + margin * 3, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + margin + (S_max(c) - min_S) / (max_S - min_S) * graphSize;
			int y2 = img.rows + margin + (S_max(c + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw w_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSize + margin * 2 + w_max(c);
			int y2 = img.rows + graphSize + margin * 2 + w_max(c + 1);

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& Hor, const cv::Mat_<float>& w_max, const std::string& filename) {
		double min_S, max_S;
		double min_Hor, max_Hor;
		double min_w, max_w;
		cv::minMaxLoc(S_max, &min_S, &max_S);
		cv::minMaxLoc(Hor, &min_Hor, &max_Hor);
		cv::minMaxLoc(w_max, &min_w, &max_w);

		int graphSize = std::max(80.0, img.rows * 0.25);
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows + graphSize * 2 + max_w + margin * 4, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + margin + (S_max(c) - min_S) / (max_S - min_S) * graphSize;
			int y2 = img.rows + margin + (S_max(c + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Hor
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSize + margin * 2 + (Hor(c) - min_Hor) / (max_Hor - min_Hor) * graphSize;
			int y2 = img.rows + graphSize + margin * 2 + (Hor(c + 1) - min_Hor) / (max_Hor - min_Hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw w_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSize * 2 + margin * 3 + w_max(c);
			int y2 = img.rows + graphSize * 2 + margin * 3 + w_max(c + 1);

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& Hor, const cv::Mat_<float>& w_max, const std::vector<float>& x_splits, const std::string& filename) {
		double min_S, max_S;
		double min_Hor, max_Hor;
		double min_w, max_w;
		cv::minMaxLoc(S_max, &min_S, &max_S);
		cv::minMaxLoc(Hor, &min_Hor, &max_Hor);
		cv::minMaxLoc(w_max, &min_w, &max_w);

		int graphSize = std::max(80.0, img.rows * 0.25);
		int margin = graphSize * 0.2;
		cv::Mat result(img.rows + graphSize * 2 + max_w + margin * 4, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// draw S_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + margin + (S_max(c) - min_S) / (max_S - min_S) * graphSize;
			int y2 = img.rows + margin + (S_max(c + 1) - min_S) / (max_S - min_S) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw Hor
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSize + margin * 2 + (Hor(c) - min_Hor) / (max_Hor - min_Hor) * graphSize;
			int y2 = img.rows + graphSize + margin * 2 + (Hor(c + 1) - min_Hor) / (max_Hor - min_Hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw w_max
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + graphSize * 2 + margin * 3 + w_max(c);
			int y2 = img.rows + graphSize * 2 + margin * 3 + w_max(c + 1);

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		// draw x splits
		for (int i = 0; i < x_splits.size(); ++i) {
			cv::line(result, cv::Point(x_splits[i], 0), cv::Point(x_splits[i], img.rows), cv::Scalar(0, 255, 255), 3);
		}

		cv::imwrite(filename, result);
	}

}
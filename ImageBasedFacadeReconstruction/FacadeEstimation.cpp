#include "FacadeEstimation.h"
#include "CVUtils.h"

namespace fe {

	void subdivideFacade(const cv::Mat& img, std::vector<float>& y_split, std::vector<float>& x_split, std::vector<std::vector<fs::WindowPos>>& win_rects) {
		// convert the image to grayscale
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);

		// vertical split
		cv::Mat_<float> SV_max;
		cv::Mat_<float> h_max;
		fs::computeSV(grayImg, SV_max, h_max, std::make_pair(10, 40));

		// estimate the floor height
		float floor_height = cvutils::getMostPopularValue(h_max, 3, 3);

		// horizontal split
		cv::Mat_<float> SH_max;
		cv::Mat_<float> w_max;
		fs::computeSH(grayImg, SH_max, w_max, std::make_pair(10, 40));

		float tile_width = cvutils::getMostPopularValue(w_max, 3, 3);

		// compute Ver(y) and Hor(x)
		cv::Mat_<float> Ver;
		cv::Mat_<float> Hor;
		fs::computeVerAndHor(img, Ver, Hor, floor_height * 0.1);

		// Facadeのsplit linesを求める
		fs::getSplitLines(Ver, y_split);
		fs::refineSplitLines(y_split);
		fs::getSplitLines(Hor, x_split);
		fs::refineSplitLines(x_split);
		//fs::outputFacadeStructure(img, y_split, x_split, filename_prefix + std::string("_subdivision.png"), 1);

		cv::Mat detected_edges;
		cv::Canny(img, detected_edges, 30, 100, 3);
		cv::imwrite("detected_edges.png", detected_edges);
		cv::Mat detected_edges_inverted;
		cv::bitwise_not(detected_edges, detected_edges_inverted);
		cv::imwrite("detected_edges_inverted.png", detected_edges_inverted);

		// facadeの端のエッジを削除する
		int margin = 8;
		for (int r = 0; r < detected_edges.rows; ++r) {
			for (int c = 0; c < detected_edges.cols; ++c) {
				if (r < margin || r >= detected_edges.rows - margin || c < margin || c >= detected_edges.cols - margin) {
					detected_edges.at<unsigned char>(r, c) = 0;
				}
			}
		}

		// 各tileの窓の位置を求める
		int window_count = 0;
		win_rects.resize(y_split.size() - 1);
		for (int i = 0; i < y_split.size() - 1; ++i) {
			win_rects[i].resize(x_split.size() - 1);
			for (int j = 0; j < x_split.size() - 1; ++j) {
				cv::Mat tile(img, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
				cv::Mat tile_edges(detected_edges, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
				if (fs::subdivideTile(tile, tile_edges, 10, 1, win_rects[i][j])) {
					window_count++;
				}
				else {
					win_rects[i][j] = fs::WindowPos();
				}
			}
		}
		std::cout << "Window count: " << window_count << std::endl;
		//fs::outputFacadeAndWindows(img, y_split, x_split, window_rects, filename_prefix + std::string("_windows.png"));

		/*
		// 窓の位置をalignする
		std::vector<std::vector<fs::WindowPos>> refined_window_rects;
		fs::refine(y_split, x_split, window_rects);
		fs::align(y_split, x_split, window_rects);
		//outputFacadeAndWindows(img, y_split, x_split, window_rects, "facade_windows_refined.png");
		*/
	}

}
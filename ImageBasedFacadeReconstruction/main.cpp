﻿/**
* 論文Image-based procedural modeling of facadesに基づいて、
* Facadeを分割する。
* 5/24に第一回の報告をしたが、いろいろ難しいことが分かった。
* まず、MIではうまくsymmetryをキャプチャできないケースがある。
*
* @author Gen Nishida
* @version 1.0
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "CVUtils.h"
#include "Utils.h"
#include <time.h>
#include "CVUtilsTest.h"
#include "visualization.h"
#include "facade_segmentation.h"

#ifndef SQR
#define	SQR(x)	((x) * (x))
#endif

using namespace std;

void clusterFloors(const cv::Mat& img, const vector<int>& y_split, int max_cluster, vector<cv::Mat>& floors, vector<int>& labels, vector<cv::Mat>& centroids) {
	floors.resize(y_split.size() - 1);
	for (int i = 0; i < y_split.size() - 1; ++i) {
		int height = y_split[y_split.size() - i - 1] - y_split[y_split.size() - i - 2];
		floors[i] = cv::Mat(img, cv::Rect(0, y_split[y_split.size() - i - 2], img.cols, height));
	}

	cvutils::clusterImages(floors, labels, centroids, max_cluster);

	cout << "Facade segmentation:" << endl;
	for (int i = 0; i < labels.size(); ++i) {
		cout << "class: " << labels[i] << endl;
	}
}

void augumentTiles(const vector<int>& labels, vector<int>& augumented_tiles, int N) {
	vector<pair<int, int>> repetition;
	for (int i = 0; i < labels.size();) {
		repetition.push_back(make_pair(labels[i], 1));
		int next = i + 1;
		for (; next < labels.size(); ++next) {
			if (labels[i] == labels[next]) {
				repetition.back().second++;
			}
			else {
				break;
			}
		}

		i = next;
	}

	if (labels.size() != N) {
		int total_repetition = 0;
		for (int i = 0; i < repetition.size(); ++i) {
			if (repetition[i].second > 1) {
				total_repetition += repetition[i].second;
			}
		}

		int total_floors = 0;
		float ratio = (float)(N - (int)labels.size()) / (float)total_repetition;
		for (int i = 0; i < repetition.size(); ++i) {
			if (repetition[i].second > 1) {
				repetition[i].second += repetition[i].second * ratio;
			}

			total_floors += repetition[i].second;
		}

		if (total_floors != N) {
			for (int i = 0; i < repetition.size(); ++i) {
				if (repetition[i].second > 1) {
					repetition[i].second += N - total_floors;
					break;
				}
			}
		}
	}

	augumented_tiles.resize(N);
	int index = 0;
	for (int i = 0; i < repetition.size(); ++i) {
		for (int j = 0; j < repetition[i].second; ++j) {
			augumented_tiles[index] = repetition[i].first;
			index++;
		}
	}
}

/**
 * ２つのラベル列の類似度を返却する。
 *
 * @param labels			推定されたラベル列
 * @param terminals			Grammarから生成されたterminal列
 * @param converted_labels	Grammarをconstraintとしてrefineしたラベル列の結果
 * @return					類似度（一致するラベル数）
 */
float compute_similarity(const vector<int>& labels, const vector<int> terminals, vector<int>& converted_labels) {
	vector<int> mapping = utils::findBestAssignment(terminals, labels);
	converted_labels.resize(labels.size());

	float sim = 0.0f;
	for (int i = 0; i < labels.size(); ++i) {
		converted_labels[i] = mapping[terminals[i]];
		if (mapping[terminals[i]] == labels[i]) sim++;
	}

	return sim;
}

vector<int> findBestFacadeGrammar(const vector<int>& labels) {
	int N = labels.size();

	vector<int> grammar_terminals(N);
	vector<int> refined_labels;
	vector<int> converted_labels;

	// facade grammar #1
	for (int i = 0; i < N; ++i) {
		grammar_terminals[i] = 0;
	}
	float max_sim = compute_similarity(labels, grammar_terminals, refined_labels);
	int best_id = 0;

	// facade grammar #2
	for (int i = 0; i < N; ++i) {
		if (i < N - 1) {
			grammar_terminals[i] = 0;
		}
		else {
			grammar_terminals[i] = 1;
		}
	}
	float sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 1;
		refined_labels = converted_labels;
	}

	// facade grammar #3
	for (int i = 0; i < N; ++i) {
		if (i == 0) {
			grammar_terminals[i] = 0;
		}
		else {
			grammar_terminals[i] = 1;
		}
	}
	sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 2;
		refined_labels = converted_labels;
	}
	
	// facade grammar #4
	for (int i = 0; i < N; ++i) {
		if (i == 0) {
			grammar_terminals[i] = 0;
		}
		else if (i < N - 1) {
			grammar_terminals[i] = 1;
		}
		else {
			grammar_terminals[i] = 2;
		}
	}
	sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 3;
		refined_labels = converted_labels;
	}

	cout << "Best grammar: " << best_id << endl;

	return refined_labels;
}

vector<int> findBestFloorGrammar(const vector<int>& labels) {
	int N = labels.size();

	vector<int> grammar_terminals(N);
	vector<int> refined_labels;
	vector<int> converted_labels;

	// floor grammar #1
	for (int i = 0; i < N; ++i) {
		grammar_terminals[i] = 0;
	}
	float max_sim = compute_similarity(labels, grammar_terminals, refined_labels);
	int best_id = 0;

	// floor grammar #2
	for (int i = 0; i < N; ++i) {
		if (i == 0) {
			grammar_terminals[i] = 0;
		}
		else if (i < N - 1) {
			grammar_terminals[i] = 1;
		}
		else {
			grammar_terminals[i] = 0;
		}
	}
	float sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 1;
		refined_labels = converted_labels;
	}

	// floor grammar #3
	int mid = N / 2;
	for (int i = 0; i < N; ++i) {
		if (i == 0) {
			grammar_terminals[i] = 0;
		}
		else if (i < mid) {
			grammar_terminals[i] = 1;
		}
		else if (i == mid) {
			grammar_terminals[i] = 2;
		}
		else if (i > mid && i < N - 1) {
			grammar_terminals[i] = 1;
		}
		else {
			grammar_terminals[i] = 0;
		}
	}
	sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 2;
		refined_labels = converted_labels;
	}

	// floor grammar #4
	for (int i = 0; i < N; ++i) {
		if (i % 2 == 0) {
			grammar_terminals[i] = 0;
		}
		else if (i < N - 1 && i % 2 != 0) {
			grammar_terminals[i] = 1;
		}
		else {
			grammar_terminals[i] = 0;
		}
	}
	sim = compute_similarity(labels, grammar_terminals, converted_labels);
	if (sim > max_sim) {
		max_sim = sim;
		best_id = 3;
		refined_labels = converted_labels;
	}

	cout << "Best grammar: " << best_id << endl;

	return refined_labels;
}

void clusterTiles(const cv::Mat& img, const vector<int>& x_split, int max_cluster, vector<cv::Mat>& tiles, vector<int>& labels, vector<cv::Mat>& centroids) {
	tiles.resize(x_split.size() - 1);
	for (int i = 0; i < x_split.size() - 1; ++i) {
		int width = x_split[i + 1] - x_split[i];
		tiles[i] = cv::Mat(img, cv::Rect(x_split[i], 0, width, img.rows));
	}

	cvutils::clusterImages(tiles, labels, centroids, max_cluster);

	/*
	cout << "Floor segmentation:" << endl;
	for (int i = 0; i < labels.size(); ++i) {
		cout << "class: " << labels[i] << endl;
	}
	*/
}

void subdivideFacade(const cv::Mat& img) {
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	//////////////////////////////////////////////////////////////////////////////
	// obtain edge image
#if 0
	cv::Mat grad;
	cv::Mat grad_x;
	cv::Mat grad_y;
	cv::Mat abs_grad_x;
	cv::Mat abs_grad_y;
	cv::Sobel(grayImg, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	cv::Sobel(grayImg, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//cv::imwrite("grad.png", grad);
	grad.convertTo(grayImg, CV_8U);
	//cv::imwrite("grad2.png", grayImg);
#endif
	//////////////////////////////////////////////////////////////////////////////

	// vertical split
	cv::Mat_<float> SV_max;
	cv::Mat_<float> h_max;
	computeSV(grayImg, SV_max, h_max, make_pair(10, 40));

	// visualize SV_max(y) and h_max(y)
	outputFacadeStructureV(img, SV_max, h_max, "SV_max.png", 1);

	// estimate the floor height
	float floor_height = cvutils::getMostPopularValue(h_max, 3, 3);
	cout << "Floor height: " << floor_height << endl;
	
	// horizontal split
	cv::Mat_<float> SH_max;
	cv::Mat_<float> w_max;
	computeSH(grayImg, SH_max, w_max, make_pair(10, 40));

	// visualize SH_max(x) and w_max(x)
	outputFacadeStructureH(img, SH_max, w_max, "SH_max.png", 1);

	float tile_width = cvutils::getMostPopularValue(w_max, 3, 3);
	cout << "Tile width: " << tile_width << endl;

	// visualize SV_max(y), h_max(y), SH_max(x) and w_max(x)
	outputFacadeStructure(img, SV_max, h_max, SH_max, w_max, "SV_SH_max.png", 1);

	// compute Ver(y) and Hor(x)
	cv::Mat_<float> Ver;
	cv::Mat_<float> Hor;
	computeVerAndHor(img, Ver, Hor, floor_height * 0.1);

	/////////////////////////////////////////////////////////////////
	// DEBUG
	cvutils::outputImageWithHorizontalAndVerticalGraph(img, Ver, Hor, "facade_subdivision_by_ver_hor.png", cvutils::LOCAL_MINIMUM, 1);
	/////////////////////////////////////////////////////////////////
	
	time_t start = clock();

	// Facadeのsplit linesを求める
	vector<int> y_split;
	//findBestHorizontalSplitLines(img, Ver, floor_height * 0.85, floor_height * 1.85, y_split);
	getSplitLines(Ver, y_split);
	refineSplitLines(y_split);
	vector<int> x_split;
	//findBestVerticalSplitLines(img, Hor, tile_width * 0.4, tile_width * 1.85, x_split);
	getSplitLines(Hor, x_split);
	refineSplitLines(x_split);
	time_t end = clock();
	cout << "Time: " << (end - start) << "msec." << endl;
	outputFacadeStructure(img, y_split, x_split, "facade_subdivision.png", 1);

	cv::Mat detected_edges;
	cv::Canny(img, detected_edges, 30, 100, 3);
	cv::imwrite("detected_edges.png", detected_edges);

	// facadeの端のエッジを削除する
	int margin = 8;
	for (int r = 0; r < detected_edges.rows; ++r) {
		for (int c = 0; c < detected_edges.cols; ++c) {
			if (r < margin || r >= detected_edges.rows - margin || c < margin || c >= detected_edges.cols - margin) {
				detected_edges.at<unsigned char>(r, c) = 0;
			}
		}
	}

#if 0
	cv::Mat detected_vertical_edges;
	cv::Mat detected_horizontal_edges;
	cv::reduce(detected_edges, detected_vertical_edges, 0, CV_REDUCE_SUM, CV_32F);
	cv::reduce(detected_edges, detected_horizontal_edges, 1, CV_REDUCE_SUM, CV_32F);

	cvutils::outputImageWithHorizontalAndVerticalGraph(detected_edges, detected_horizontal_edges, detected_vertical_edges, "detected_edges_votes.png");
	
	cv::Mat vertical_edge_max;
	cv::Mat horizontal_edge_max;
	cv::reduce(detected_vertical_edges, vertical_edge_max, 1, CV_REDUCE_MAX, CV_32F);
	cv::reduce(detected_horizontal_edges, horizontal_edge_max, 0, CV_REDUCE_MAX, CV_32F);
#endif
	
	// 各tileの窓の位置を求める
	vector<vector<cv::Rect>> window_rects;
	int window_count = 0;
	window_rects.resize(y_split.size() - 1);
	for (int i = 0; i < y_split.size() - 1; ++i) {
		window_rects[i].resize(x_split.size() - 1);
		for (int j = 0; j < x_split.size() - 1; ++j) {
			cv::Mat tile(img, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
			cv::Mat tile_edges(detected_edges, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
			if (subdivideTile(tile, tile_edges, 10, 1, window_rects[i][j])) {
				window_count++;
			}
			else {
				window_rects[i][j] = cv::Rect(0, 0, 0, 0);
			}
		}
	}
	cout << "Window count: " << window_count << endl;
	outputFacadeAndWindows(img, y_split, x_split, window_rects, "facade_windows.png");

	// 窓の位置をalignする
	refine(y_split, x_split, window_rects);
	outputFacadeAndWindows(img, y_split, x_split, window_rects, "facade_windows_refined.png");

	// 各floorのsimilarityを計算する
	vector<cv::Mat> floors;
	vector<cv::Mat> floor_centroids;
	vector<int> floor_labels;
	clusterFloors(img, y_split, 4, floors, floor_labels, floor_centroids);
	outputFacadeSegmentation(img, y_split, floor_labels, "facade_labeled.png");

	// 最も類似するfacade grammarを探す
	floor_labels = findBestFacadeGrammar(floor_labels);
	outputFacadeSegmentation(img, y_split, floor_labels, "facade_labeled_refined.png");

	// 各tileのsimilarityを計算する
	{
		vector<vector<int>> all_labels(floor_labels.size());
		for (int i = 0; i < all_labels.size(); ++i) {
			all_labels[i].resize(x_split.size() - 1);
		}

		vector<cv::Mat> tiles;
		vector<cv::Mat> tile_centroids;
		vector<int> tile_labels;
		int label_offset = 0;
		for (int i = 0; i < floor_centroids.size(); ++i) {
			clusterTiles(floor_centroids[i], x_split, 4, tiles, tile_labels, tile_centroids);

			// このクラスタに属する全てのフロアに、計算されたlabelをコピーする
			for (int k = 0; k < all_labels.size(); ++k) {
				if (floor_labels[k] != i) continue;

				for (int l = 0; l < tile_labels.size(); ++l) {
					all_labels[k][l] = tile_labels[l] + label_offset;
				}
			}

			label_offset += tile_centroids.size();
		}
		outputFloorSegmentation(img, y_split, x_split, all_labels, "floor_labeled.png");
	}

	// 各種のfloorについて、最も類似するfloor grammarを探す
	{
		vector<vector<int>> all_labels(floor_labels.size());
		for (int i = 0; i < all_labels.size(); ++i) {
			all_labels[i].resize(x_split.size() - 1);
		}

		vector<cv::Mat> tiles;
		vector<cv::Mat> tile_centroids;
		vector<int> tile_labels;
		int label_offset = 0;
		for (int i = 0; i < floor_centroids.size(); ++i) {
			clusterTiles(floor_centroids[i], x_split, 4, tiles, tile_labels, tile_centroids);

			tile_labels = findBestFloorGrammar(tile_labels);

			// このクラスタに属する全てのフロアに、計算されたlabelをコピーする
			for (int k = 0; k < all_labels.size(); ++k) {
				if (floor_labels[k] != i) continue;

				for (int l = 0; l < tile_labels.size(); ++l) {
					all_labels[k][l] = tile_labels[l] + label_offset;
				}
			}

			label_offset += tile_centroids.size();
		}
		outputFloorSegmentation(img, y_split, x_split, all_labels, "floor_labeled_refined.png");
	}
}

int main() {
	cvutils::test_cvutils();

	//cv::Mat img = cv::imread("../facade_small/facade2.png");
	cv::Mat img = cv::imread("\\\\matrix.cs.purdue.edu\\cgvlab\\gen\\meeting\\2016\\20160531\\facade_images\\facade1.png");

	subdivideFacade(img);

	return 0;
}

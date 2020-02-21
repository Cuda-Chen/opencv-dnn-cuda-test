#include <iostream>
#include <string>
#include <ctime>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{
	double confidenceThreshold = 0.5;
	double NMSThreshold = 0.4;
	int numClasses = 80;
	clock_t start, end;

	string modelPath = "../yolos/yolov3.weights";
	string configPath = "../yolos/yolov3.cfg";
	string inputImg = "../yolos/dog.jpg";
	Net net = readNet(modelPath, configPath);
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);

	Mat image = imread(inputImg);
	Mat blob = blobFromImage(image, 0.00392, Size(416, 416), Scalar(), true, false, CV_32F);
	auto outputNames = net.getUnconnectedOutLayersNames();
	vector<Mat> detections;

	// warmup
	for(int i = 0; i < 3; i++)
	{
		net.setInput(blob);
		net.forward(detections, outputNames);
	}

	// benckmark
	start = clock();
	for(int i = 0; i < 100; i++)
	{
		net.setInput(blob);
		net.forward(detections, outputNames);
	}
	end = clock();

	double timeTaken = double(end - start) / double(CLOCKS_PER_SEC);
	cout << "Time per inference: " << timeTaken << " seconds" << endl
		<< "FPS: " << 1.0 / timeTaken << endl;

	return 0;
}

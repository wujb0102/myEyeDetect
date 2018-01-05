#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

int main()
{
	VideoCapture cap(0); 	//打开默认摄像头
	if(!cap.isOpened()){
		return -1;
	}
	Mat frame;
	Mat edges;
	Mat gray;

	CascadeClassifier cascade;
	bool stop = false;
	cascade.load("haarcascade_frontalface_default.xml");
	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
	model->read("MyFaceLBPHModel.xml");
	while(1){
		cap >> frame;
		//建立用于存放人脸的向量容器
		vector<Rect> faces(0);
		cvtColor(frame, gray, CV_BGR2GRAY);
		//改变图像大小，使用双线性插值
		//resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
		//变换后的图象进行直方图均值化处理
		equalizeHist(gray, gray);
		//cascade.detectMultiScale(gray, faces, 1.07, 20, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		cascade.detectMultiScale(gray, faces, 1.12, 6, 0, Size(100, 100), Size(500, 500));
		Mat face;
		Point text_lb;
		for(size_t i = 0; i < faces.size(); i++){
			if(faces[i].height > 0 && faces[i].width > 0){
				face = gray(faces[i]);
				text_lb = Point(faces[i].x, faces[i].y);
			}
		}
		Mat face_test;
		int predictLabel = 0;
		double number = 0;
		if(face.rows >= 120){
			resize(face, face_test, Size(92, 112));
		}
		if(!face_test.empty()){
			model->predict(face_test, predictLabel, number);
		}
		cout<< "predict:" << predictLabel << endl;
		cout<< "number:" << number << endl;
		if(predictLabel == 41 && number < 90){
			string name = "WuJiabin";
		       	putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));	
		}
		imshow("face", frame);
		waitKey(200);
	}
	return 0;
}

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	CascadeClassifier cascade;	//创建级联分类器对象
	cascade.load("haarcascade_frontalface_default.xml"); 	//加载分类模型的文件
	VideoCapture cap; 	//创建摄像头对象
	cap.open(0); 	//通过对象调用摄像头类的方法，打开默认摄像头
	Mat frame;	//创建Mat对象，用来存放摄像头采集的原始图片，即视频中的一帧
	int pic_count = 0;
	int pic_num = 1;
	while(1){
		if(pic_count == 1000)pic_count = 0;
		cap >> frame; 	//截取视频中的一帧
		pic_count++;
		if((pic_count % 10) == 0){
			std::vector<Rect> faces; 	//创建容器，用来存放一张图片中检测出来的人脸框
			Mat frame_gray; 	//创建一个Mat对象，用来存放待检测图片的灰度化图片
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY); 	//把视频中的一帧转化为灰度图片
			equalizeHist(frame_gray,frame_gray);  //直方图均衡行
			cascade.detectMultiScale(frame_gray, faces, 1.08, 10, 0, Size(100, 100), Size(500, 500)); 	//进行人脸检测,把一张图片中的所有的人脸框对象保存在rect容器中
			for(size_t i = 0; i < faces.size(); i++){ 	//在图片中框出人脸
				rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
			}
			//当图片中只有一张人脸，再保存下来。一张图片出现多张人脸就丢弃这张图片。
			if(faces.size() == 1){
				Mat faceROI = frame_gray(faces[0]); 	//创建Mat对象临时保存检测出来的人脸
				Mat myFace; 	//创建Mat对象保存图像缩放后的图像
				resize(faceROI, myFace, Size(92, 112)); 	//把图片缩放成固定大小，用于人脸识别的训练
				putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 255), 2, LINE_AA); //在图片框中显示文字
				string filename = format("./att_faces/s41/%d.jpg",pic_num); 	//创建string对象保存文件路径和文件目录
				imwrite(filename, myFace); 	//将文件写到指定位置
				imshow(filename, myFace); 	//显示图片
				waitKey(1); 	//等待按键按下，没有这行图片显示会存在问题
				//destroyWindow(filename); 	//销毁窗口
				pic_num++;
				if(pic_num == 50){
					return 0;
				}
			imshow("frame",frame);
			waitKey(500);

			}
		}
		imshow("frame",frame);
		waitKey(1);
	}
	return 0;
}


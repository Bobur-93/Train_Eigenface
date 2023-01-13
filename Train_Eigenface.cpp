// Train_Eigenface.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include "opencv2\core.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/face.hpp"

using namespace cv;
using namespace std;
using namespace cv::face;




Mat Cut_func(int i, int j)
{
	int t, d, e, index;
	Mat image;
	//CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	//CascadeClassifier profile_face("haarcascade_profileface.xml");

	//vector<Rect> faces;
	//vector<Rect> faces_1;

	//24 - 26
	string path_train;
	string ost = ").jpg";

	if (i==0)
	{
		path_train = "E:\\Eigen train\\I_1\\image (1";
		index = 26;
	}
	else
	{
		if (i==1)
		{
			path_train = "E:\\Eigen train\\Azim-aka_1\\image (1";
			index = 33;			
		}
		else
		{
			path_train = "E:\\Eigen train\\Aziz_1\\image (1";
			index = 29;			
		}
	}
	if (j < 10)
	{
		path_train[index] = j + '0';
		path_train = path_train + ost;
		cout << path_train << '\n';
		//image = imread(path_train, IMREAD_GRAYSCALE);
	}
	else
	{
		if (j < 100)
		{
			d = j / 10;
			e = j % 10;
			path_train[index] = d + '0';
			path_train.push_back(e + '0');
			path_train = path_train + ost;
			cout << path_train << '\n';
			//image = imread(path_train, IMREAD_GRAYSCALE);
		}
		else
		{
			t = j / 100;
			d = j % 100;
			d = d / 10;
			e = j % 100;
			e = e % 10;
			path_train[index] = t + '0';
			path_train.push_back(d + '0');
			path_train.push_back(e + '0');
			path_train = path_train + ost;
			cout << path_train << '\n';
			//image = imread(path_train, IMREAD_GRAYSCALE);
		}
	}

	image = imread(path_train, IMREAD_GRAYSCALE);
	/*if (image.data == NULL)
	{
		return 1;
	}
	faceDetector.detectMultiScale(image, faces);

	for (int l = 0;  l < faces.size();  l++)
	{
		new_image = image(faces[l]);
	}*/
	
	//imshow("image", new_image);
	//waitKey();

	

	return image;
}

int main()
{
	int i, j, l, h, max_h=0, max_w=0, test_label;

	

	vector<Mat> images;
	Mat frame, blurred, test_image;
	vector<int> labels;

	string path = "E:\\Eigen train\\1_01.jpg";
	

	

	j = 1;
	l = 0;
	h = 0;

	for ( i = 0; i < 3; i++)
	{
		for ( j = 1; j < 121; j++)
		{
			frame = Cut_func(i, j);
			images.push_back(frame);
			labels.push_back(i);
		}
	}

	for (i = 0; i < images.size(); i++)
	{
		resize(images[i], images[i], Size(300, 400), 1.0, 1.0, INTER_CUBIC);
	}

	test_image = images[images.size() - 1];
	test_label = labels[labels.size() - 1];

	images.pop_back();
	labels.pop_back();
	

	//cout << path[15] << ' ' << path[17] << '\n';
	/*for (i = 0; i < 3; i++)
	{		
		if (i == 0)
		{
			for (int q = 1; q < 30; q++)
			{
				path[15] = '1';
				path[17] = (q / 10) + '0';
				path[18] = (q % 10) + '0';
				//cout << path << '\n';
				frame = imread(path, IMREAD_GRAYSCALE);
				images.push_back(frame);
				labels.push_back(i);
				/*for (int j = 1; j < 121; j = j + 6)
				{
					GaussianBlur(frame, blurred, Size(j, j), 0, 0);
					cv::blur(blurred, blurred, Size(j, j));
					images.push_back(blurred);
					labels.push_back(i);
				}*/	
			/* }
		}
		else
		{
			if (i == 1)
			{
				for (int q = 1; q < 26; q++)
				{
					path[15] = '2';
					path[17] = (q / 10) + '0';
					path[18] = (q % 10) + '0';
					//cout << path << '\n';
					frame = imread(path, IMREAD_GRAYSCALE);
					images.push_back(frame);
					labels.push_back(i);
					/*for (int j = 1; j < 121; j = j + 6)
					{
						GaussianBlur(frame, blurred, Size(j, j), 0, 0);
						cv::blur(blurred, blurred, Size(j, j));
						images.push_back(blurred);
						labels.push_back(i);
					}*/
				/* }
			}
			else
			{
				for (int q = 1; q < 26; q++)
				{
					path[15] = '3';
					path[17] = (q / 10) + '0';
					path[18] = (q % 10) + '0';
					//cout << path << '\n';
					frame = imread(path, IMREAD_GRAYSCALE);
					images.push_back(frame);
					labels.push_back(i);
					/*for (int j = 1; j < 121; j = j + 6)
					{
						GaussianBlur(frame, blurred, Size(j, j), 0, 0);
						cv::blur(blurred, blurred, Size(j, j));
						images.push_back(blurred);
						labels.push_back(i);
					}*/
				/* }
			}
		}
		
		/*path[15] = j + '0';
		path[17] = l + '0';
		path[18] = h + '0';*/
		//cout << path << '\n';
		//frame = imread(path, IMREAD_GRAYSCALE);
		//imshow("image", frame);
		//if (waitKey(1) == 27)
		//	break;
		//images.push_back(frame);
		//labels.push_back(j);
	//}

	//base_train(images[0],0);

	//cout << images.size() << '\n';

	/*for (i = 0; i < images.size(); i++)
	{
		imshow("image", images[i]);
		if (waitKey(1) == 27)
			break;
	}*/

	//imshow("image", images[15]);
		

	Ptr<EigenFaceRecognizer> eigen_model = EigenFaceRecognizer::create(120);
	Ptr<FaceRecognizer>  fisher_model = FisherFaceRecognizer::create(120);
	Ptr<LBPHFaceRecognizer> lbph_model = LBPHFaceRecognizer::create(120);

	//train data
	eigen_model->train(images, labels);
	eigen_model->save("E:\\Eigen train\\eigenface.yml");
	cout << "Training finished....\n" << endl;

	fisher_model->train(images, labels);
	fisher_model->save("E:\\Eigen train\\Fisherfaces.yml");
	cout << "Training finished....\n" << endl;

	lbph_model->train(images, labels);
	lbph_model->save("E:\\Eigen train\\Lbph_faces.yml");
	cout << "Training finished....\n" << endl;

	int predictedLabel = eigen_model->predict(test_image);

	cout << "actual label : " << test_label << "\n predict label : " << predictedLabel << '\n';

	return 0;
}

//visual odometry trial-1
#include<opencv2/opencv.hpp>
//#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include"vo_features.h"
//#include<string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


#define MAX_FRAME 1000
#define MIN_NUM_FEAT 200

Mat K = (Mat_<double>(3,3) << 1300.916421, 0.000000, 693.632446, //left camera matrix
                          0.000000, 1236.693231, 263.159396, 
                          0.000000, 0.000000, 1.000000);

Mat M = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0, 
                              0.0, 0.0, 1.0, 0.0);
 //Mat P1(3,4, CV_32FC1);
 //Mat P2(3,4, CV_32FC1);

//cv:: void gemm(K,M,P1);
// IMP: Change the file directories (4 places) according to where your dataset is saved before running!
 cv:: FileStorage fsrc("pnts_3D.yml",FileStorage::WRITE);

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{
  
  string line;
  int i = 0;
  ifstream myfile ("00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}

int main()
{
 double scale = 1.0;
 char filename1[200];
 char filename2[200];
    Mat img1;
    Mat img2;
   Mat R_f;
Mat t_f;
sprintf(filename1, "img_%d.jpg", 0);
  sprintf(filename2, "img_%d.jpg", 1);
    img1 = imread(filename1);
    img2 = imread(filename2); 
cout<<img1.size()<<endl; 
   if(!img1.data||!img2.data)
    {
        cout<<"images not read"<<endl;
    }
char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  cv::Point textOrg(10, 50);
         
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keypoints_1,keypoints_2;
    detector->detect(img1,keypoints_1);
    detector->detect(img2,keypoints_2);
    
    //Mat descriptor_1, descriptor_2;
    //Ptr<DescriptorExtractor> Descriptor = ORB::create();
    //Descriptor->compute(img1,keypoints_1,descriptor_1);
    //Descriptor->compute(img2,keypoints_2,descriptor_2);
    //imshow("desc_1",descriptor_1);  
    //conversion from keypoints to points for tracking
    
    vector<Point2f> points_1,points_2;
    //vector<Point3f>triangle3d;
    
    
    KeyPoint::convert(keypoints_1,points_1);  
    KeyPoint::convert(keypoints_2,points_2);
    //cout<<points_1.size()<<endl;
    //cout<<points_2.size()<<endl; 
  //  feature tracking
    //using optical flow with KLT
 //features have been tracked using Optical FLow with KLT tracker.
  
  vector<uchar> status;
    vector<float> err;
    Size winSize = Size(21,21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.01);
    calcOpticalFlowPyrLK(img1,img2,points_1,points_2,status,err,winSize,3,termcrit,0,0.001);
    int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points_2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
    	  points_1.erase (points_1.begin() + (i - indexCorrection));
     		  points_2.erase (points_2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }
   //cout<<status.size()<<endl;

double f =  1300.916421 ;
Point2d pp(693.632446, 263.159396);
//cv::Mat pnts3D(4,points_1.size(),CV_64F);
Mat E,R,t, mask;
E = findEssentialMat(points_2, points_1, f, pp, RANSAC, 0.999, 1.0, mask);
recoverPose(E, points_2, points_1, R, t, f, pp, mask);
cout<<R.size()<<endl;
cout<<t.size()<<endl;
cout<<"pose recovered"<<endl;
 Mat P1 = K * M;
 cout<<P1<<endl;
cv::vconcat(R,t.t(),M);
cout<<M<<endl;
cout<<M.size()<<endl;
cout<<"concatenated"<<endl;
cout<<K.size()<<endl;
Mat P2 = K*M.t();
cout<<"P2 is"<<P2<<endl;
Mat pnts3D(4,points_1.size(),CV_64F);
Mat pnts3D_vis(3,points_1.size(),CV_64F);
triangulatePoints(P1,P2,points_1,points_2, pnts3D);
//fsrc<<"feature point coordinates-"<<pnts3D.t();
for(int i = 0; i<pnts3D.cols; i++)
{
  pnts3D_vis.at<double>(0,i) = pnts3D.at<double>(0,i)/pnts3D.at<double>(3,i);
  pnts3D_vis.at<double>(1,i) = pnts3D.at<double>(1,i)/pnts3D.at<double>(3,i);
  pnts3D_vis.at<double>(2,i) = pnts3D.at<double>(2,i)/pnts3D.at<double>(3,i);

}
Mat pnts3d;
pnts3d = pnts3D_vis.t();
fsrc<<"feature point coordinates-"<<pnts3d;
imshow("triangulated",pnts3d);
waitKey(0);
cout<<pnts3D.size()<<endl;
//cout<<"Essential Matrix"<<E<<endl;
//cout<<"Rotational"<<R<<endl;
//cout<<"Translational"<<t<<endl;
P1 = P2.clone();
Mat prevImage = img2;
  Mat currImage;
  vector<Point2f> prevFeatures = points_2;
  vector<Point2f> currFeatures;
  

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
  	sprintf(filename, "img_%d.jpg", numFrame);
cout<<filename<<endl;    
//cout << numFrame << endl;
  	Mat currImage = imread(filename);
  	//cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<uchar> status;
  	featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

  	E = findEssentialMat(currFeatures, prevFeatures, f, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, f, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


   for(int i=0;i<prevFeatures.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    }
       
  	scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

    cout << "Scale is " << scale << endl;
    //cout<<currPts<<endl;
    if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

      t_f = t_f + scale*(R_f*t);
      R_f = R*R_f;
      cv::vconcat(R_f,t_f.t(),M);
      P2 = K*M.t();
      
    triangulatePoints(P1,P2,points_1,points_2, pnts3D);
//fsrc<<"feature point coordinates-"<<pnts3D.t();
for(int i = 0; i<pnts3D.cols; i++)
{
  pnts3D_vis.at<double>(0,i) = pnts3D.at<double>(0,i)/pnts3D.at<double>(3,i);
  pnts3D_vis.at<double>(1,i) = pnts3D.at<double>(1,i)/pnts3D.at<double>(3,i);
  pnts3D_vis.at<double>(2,i) = pnts3D.at<double>(2,i)/pnts3D.at<double>(3,i);

}
Mat pnts3d;
pnts3d = pnts3D_vis.t();
fsrc<<"feature point coordinates-for frame is"<<pnts3d;
    
    P1 = P2.clone();
    
    
    }
  	
    else {
     //cout << "scale below 0.1, or incorrect translation" << endl;
    }
    
   // lines for printing results
   // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

  // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
 	  if (prevFeatures.size() < MIN_NUM_FEAT)	{
      //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
      //cout << "trigerring redection" << endl;
 		  featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

 	  }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 300;
    int y = int(t_f.at<double>(2)) + 100;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    imshow( "Road facing camera", currImage );
    imshow( "Trajectory", traj );

    waitKey(1);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;
  //cout << R_f << endl;
  //cout << t_f << endl;
return(0);
}

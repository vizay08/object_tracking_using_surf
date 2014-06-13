// obkject recognition using surf opencv

#include<opencv2\opencv.hpp>
#include<iostream>

int main()
{
	cv::Mat src,bsrc,des,bdes,result,des_src,des_des;
	char *s = (char*)malloc(sizeof(char)*40);
	cout<<"enter source image path "<<endl;
	gets(s);
	src = cv::imread(s);

	cv::cvtColor(src,bsrc,CV_BGR2GRAY);


	cout<<"enter target image path"<<endl;
	gets(s);
	des = cv::imread(s);

	

	cv::cvtColor(des,bdes,CV_BGR2GRAY);

	vector<cv::KeyPoint> srcKeypoints,desKeypoints;

	cv::SurfFeatureDetector detector(500);

	detector.detect(bsrc,srcKeypoints); //detect the keypoints in src image

	detector.detect(bdes,desKeypoints); // detect the keypoints in dest image

	cv::SurfDescriptorExtractor extractor;

	extractor.compute(bsrc ,srcKeypoints, des_src);
	
	//////////////////////////////////////////////////////////////

	extractor.compute(bdes,desKeypoints,des_des);

	cv::FlannBasedMatcher matcher;
	vector<vector<cv::DMatch>> matches;

	vector<cv::DMatch> good_matches;

	matcher.knnMatch(des_src,des_des,matches,2);


	 for(int i = 0; i < min(des_des.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {
            if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }


	 cv::Mat imageMatch;

	 vector<cv::Point2f> obj_corners(4),scene_corners(4);

    //Get the corners from the object
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( src.cols, 0 );
    obj_corners[2] = cvPoint( src.cols, src.rows );
    obj_corners[3] = cvPoint( 0, src.rows );

	 vector<cv::Point2f> obj,scene;
	 cv::Mat objImage,sceneImage ;

	 cv::Mat objCornerImage = cv::Mat(obj_corners);
	 cv::Mat sceneCornerImage = cv::Mat(scene_corners);


	 drawMatches( src, srcKeypoints, des, desKeypoints, good_matches, imageMatch, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        if (good_matches.size() >= 4)
        {
            for( int i = 0; i < good_matches.size(); i++ )
            {
                //Get the keypoints from the good matches
                obj.push_back( srcKeypoints[ good_matches[i].queryIdx ].pt );
			

				scene.push_back( desKeypoints[ good_matches[i].trainIdx ].pt );
			

            }


			objImage = cv::Mat(obj);
			sceneImage = cv::Mat(scene);
			
			cv::Mat H = cv::findHomography(objImage,sceneImage,CV_RANSAC);

            cv::perspectiveTransform( objCornerImage, sceneCornerImage, H);
			

            //Draw lines between the corners (the mapped object in the scene image )
           cv::line( imageMatch, scene_corners[0] + cv::Point2f( src.cols, 0), scene_corners[1] + cv::Point2f( src.cols, 0), cv::Scalar(0, 255, 0), 4 );
            cv::line( imageMatch, scene_corners[1] + cv::Point2f( src.cols, 0), scene_corners[2] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( imageMatch, scene_corners[2] + cv::Point2f( src.cols, 0), scene_corners[3] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( imageMatch, scene_corners[3] + cv::Point2f( src.cols, 0), scene_corners[0] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        }

	

	cv::imshow("matched image",imageMatch);

	cv::waitKey();

	return 0;
}
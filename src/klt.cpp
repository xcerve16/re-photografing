// C++
#include <iostream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <cv.h>

using namespace cv;
using namespace std;

string video_read_path = "resource/video/video2.mp4";

int main(int argc, char **argv) {

    string winName = "Test";
    vector<Point2f> featuresPrevious;
    vector<Point2f> featuresCurrent;

    VideoCapture cap;
    cap.open(video_read_path);

    Mat frame, frame_vis, cImage, imgRef, lastImgRef;

    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    bool isFirstImage = true;
    while (cap.read(frame) && waitKey(30) != 27) {

        frame_vis = frame.clone();

        cvtColor(frame_vis, cImage, CV_BGR2GRAY );
        cImage.convertTo(imgRef,  CV_8U);
        featuresPrevious = std::move(featuresCurrent);
        goodFeaturesToTrack(imgRef, featuresCurrent, 30, 0.01, 30);
        if (!imgRef.data) {
            return 1;
        }

        if (!isFirstImage) {
            vector<Point2f> featuresNextPos;
            vector<uchar> featuresFound;
            Mat err;
            calcOpticalFlowPyrLK(lastImgRef, imgRef, featuresPrevious, featuresNextPos, featuresFound, err);
            for (size_t i = 0; i < featuresNextPos.size(); i++) {
                if (featuresFound[i]) {
                    line(imgRef, featuresPrevious[i], featuresNextPos[i], Scalar(0, 0, 255), 5);
                }
            }

        }
        lastImgRef =imgRef.clone();
        imshow(winName, imgRef);
        isFirstImage = false;
        waitKey(1000 / 60);
    }

    waitKey(0);
    return 0;
}
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaarithm.hpp>
#include <unistd.h>

#define DEFAULT_DELAY_FRAME_COUNT 3
#define DEFAULT_THRESH_PIXEL 25
#define DEFAULT_MIN_SIZE_FOR_MOVEMENT 4000
#define DEFAULT_MOTION_DETECTED_PERSISTANCE 5

using namespace cv;
using namespace std;
 
int main(int argc, char **argv) {
    
    CommandLineParser cmd(argc, argv, 
            "{ mode      | 0 | 0 - camera, 1 - video}"
            "{ v video   | /home/namdz/vinbigdata/data/dreamcity_lpr/test2.mp4 | specify input video}"
            "{ g gpu     | true | GPU or CPU}"
            "{ threshold | 5.0 | Threshold for magnitude}"
            "{ threshold_pixel | 200 | Threshold for number of pixel for detect motion or not}"
            );

    cmd.about("Farneback's optical flow samples.");
    bool gpuMode = cmd.get<float>("gpu");
    int mode     = cmd.get<int>("mode");
    string pathVideo = cmd.get<string>("video");
    
    VideoCapture cap;
    if(mode == 0){
        cap.open(0); // Read from camera;
    }else{
        if(pathVideo.empty()){
            cerr << "Path video cannot empty when mode = 1\n";
            return -1;
        }
        cap.open(pathVideo);
    }
    if(!cap.isOpened()){
        cerr << "Cannot open camera or video\n";
        return -1;
    }

    Mat frame, framegray, frameDelta, thresh, firstFrame;
    vector<vector<Point> > cnts;
    
    //set the video size to 512x288 to process faster
    cap.set(3, 512);
    cap.set(4, 288);

    sleep(3);
    cap.read(frame);

    //convert to grayscale and set the first frame
    cvtColor(frame, firstFrame, COLOR_BGR2GRAY);
    GaussianBlur(firstFrame, firstFrame, Size(21, 21), 0);

    int delay_count = 0, persistent_motion_count = 0;
    int64 t0, t1;
    bool motion_detected = false;
    string status;

    while(cap.read(frame)) {
    
        cvtColor(frame, framegray, COLOR_BGR2GRAY);        

        delay_count++;
        if(delay_count >= DEFAULT_DELAY_FRAME_COUNT){
            firstFrame = framegray.clone();
            delay_count = 0;
        }

        t0 = getTickCount();
        absdiff(firstFrame, framegray, frameDelta);
        threshold(frameDelta, thresh, DEFAULT_THRESH_PIXEL, 255, THRESH_BINARY);

        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);
        findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        motion_detected = false;
        for(int i = 0; i < cnts.size(); i++){
            double area = contourArea(cnts[i]);
            if(area > DEFAULT_MIN_SIZE_FOR_MOVEMENT){
                motion_detected = true;
                break;
            }
        }

        t1 = getTickCount();
        
        if(motion_detected){
            persistent_motion_count = DEFAULT_MOTION_DETECTED_PERSISTANCE;
        }

        if(persistent_motion_count > 0){
            persistent_motion_count--;
            motion_detected = true;
            status = "Motion detected";
        }else{
            motion_detected = false;
            status = "Non Motion";
        }
        
        stringstream s;
        s << "FPS " << cvRound(getTickFrequency()/(t1 - t0));
        
        putText(frame, status, Point(10, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 255), 2);
        putText(frame, s.str(), Point(10, 65), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 255), 2);

        imshow("Motion detector", frame);

        if(waitKey(1) == 27){
            //exit if ESC is pressed
            break;
        }
    }
    return 0;
}
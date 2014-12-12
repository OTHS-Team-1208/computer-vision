//Hot Goal Detection - The First Vision Processing System for Team 1208 - Chief Programmer Yash Patel was the FIRST CHIEF EVER TO CREATE A WORKING AND APPLICABLE VISION PRODUCT

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "libfreenect_cv.h"
#include <libfreenect.hpp>
//#include "libfreenect_cv2.h" //ONLY INCLUDE FOR ODROID

#include "timer.h"

#include <QtNetwork/QtNetwork>

#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <thread>
#include <unistd.h>

using namespace cv;
using namespace std;


// Global Constants
//Sadly, the number of threads has to be set before building and cannot be modified via file settings later. But it is an easy fix. Just recompile after changing the 4 (or whatever the ThreadNumber is) to whatever number of threads you wish to have executed.
//_____________________________________
#define NumberofThreads 4
string TargetIPAddress = "10.12.8.2"; QString QstringTargetIPAddress=QString::fromStdString(TargetIPAddress);
short TargetPortNumber = 80, HostPortNumber=7755;
bool textconsole = false;
bool GUIMODE = false;
bool DemoMode = false; unsigned int DemoThresholdingSleep=100, DemoBeginFinishSleep=1000;
unsigned int MaxAllowableSquareArea=3000, MinAllowableSquareArea=400;
double PolygonApproximationAccuracyProportion=0.020; //Used to be 0.015
double SquareCentersDistanceLimit=10;
double MaxAllowableAngle=0.30;
const string GUIWindowName = "Hot Goal Detection";
int CannyUpperLimitThreshold = 50, NumberofThresholdingIterations = 50; //N=Number of Thresholds
unsigned long int ProgramIterationLimit=0;
string LabViewKillString="Stop_Odroid";
unsigned short ReceivePacketsDelay=10;
bool TogglePyramidUpDownBlur=true, enablenetworking=true;
//_____________________________________
// End of Global Constants


// Global Variables
//_____________________________________
vector< vector < Point > > Squares, SquareInterMediate;
vector<vector< vector < Point > > > SquaresTempforMultiThreading(NumberofThreads);
Mat AbsoluteOriginalImage;
vector<Point2f> MassCenters, TemporaryCenters;
double MilliSecondsPerRenderedFrames=0;
bool IsFindSquaresBusy=false, EndProgram=false; //for LabView to tell the algorithm when to stop.
unsigned int FindSquares_Period=10;
//_____________________________________
//End of Global Variables

//_____________________________________
void FindSquares(); //Finds and tests all contours found within an image to see if it is an eligible square
    void FindSquaresThreader(int startingindex, Mat gray0, Mat gray); //This is where the FindSquares Algorithm is MultiThreaded
    void CombineThreadingData(); //This Function combines all the squares that each individual thread found
    double MaxAngleFunction( Point pt1, Point pt2, Point pt0 ); //This is function that helps to test countours for squares eligibility
void RemoveIdenticalSquares(); //Does as the Name Implies ... It removes Duplicate Squares
void UserInterfaces(const unsigned int iteration); //This is where all the user interfaces are displayed/rendered/etc.
    void SendNetworkPackets(unsigned int i, unsigned int j, unsigned int k); //Sends the Data Over UDP to the LabView Side on the Robot
    void ReceiveNetworkPackets(); //Receives Data from the Robot side LabView (In this case, Data for when to shut down algorithm)
    double SquareArea(); //Returns the Area of a Square
void DisplayGlobals(); //Displays Variable Settings and other important Settings Info
    void ReWriteGlobals(int argc, char** argv); //OverWrites Defaults Values with Values provided from the PreRunSettings.txt text file
    string truefalse (bool condition); //Returns "true" if 1 or "false" if 0;
    inline bool file_exist_check (const std::string& name); //checks whether the file arguement passed from the shell actually exists
//_____________________________________

//_____________________________________
int main(int argc, char** argv) {
    ReWriteGlobals(argc, argv); //First Step ... Check if a file argument was provided and then rewrite global constants with file values
    DisplayGlobals(); //Display those globals with std::cout
    if (GUIMODE || DemoMode) namedWindow(GUIWindowName); //If a GUI was requested, then create a GUI Window
    DECLARE_TIMING(timer); DECLARE_TIMING(findSquaresTimer); //Declare Timers for Overall MSperFrame and the time FindSquares took
    if (enablenetworking) {
    thread receivepacketthread = thread(ReceiveNetworkPackets); receivepacketthread.detach(); //detaches the receivepackets function thread
    }
    thread SendNetworkPacketsThread; //thread for Sending Packets over UDP (Used Later)
    
    for(unsigned int iteration=1 ; iteration!=ProgramIterationLimit && !EndProgram; iteration++) //If either it reaches the iteration limit or if LabView sends the EndProgram command, then the program will shutdown
    {
        START_TIMING(timer); //Start Timing for msPerFrame
        AbsoluteOriginalImage = freenect_sync_get_ir_cv(0); //Obtain IR Image from the kinect
        if( AbsoluteOriginalImage.empty() ) { //If the image wasn't loaded correctly, output an error and end the program.
            cout << "Couldn't load Image!!!! MAYBE THE DEVICE IS NOT CONNECTED!!!!" <<endl;
            return -1;
        }
        if(DemoMode) {imshow(GUIWindowName, AbsoluteOriginalImage); waitKey(DemoBeginFinishSleep);} //For Demo Mode, display and pause on original image
        START_TIMING(findSquaresTimer); FindSquares(); STOP_TIMING(findSquaresTimer); FindSquares_Period=GET_TIMING(findSquaresTimer); //Timing function calls pre and post calling the Find Squares function
        RemoveIdenticalSquares(); //Removes Identical Squares since the algorithm will find the same square multiple times since it uses multiple iterations of thresholding on the same image for higher accuracy

        if(GUIMODE || textconsole || DemoMode) {UserInterfaces(iteration);} //This is where the UserInterfaces Function is called if any UI was requested
        
        STOP_TIMING(timer); MilliSecondsPerRenderedFrames=GET_TIMING(timer); //More Timing Function Calls ...
        
        if(enablenetworking) {SendNetworkPacketsThread=thread(SendNetworkPackets, Squares.size(), MilliSecondsPerRenderedFrames, HostPortNumber); SendNetworkPacketsThread.detach();}
        //sends data to labview over UDP //You could also network the variable masscenters (which will report the center point of each square) for AutoAim type algorithms - I did not this year because it was not required.
    }
    sleep(1); //I added this to allow other threads such as the fucntion receivepackets time to finish their execution
    return 0; //return 0 upon success and end execution of the program
}
//_____________________________________

void FindSquares() { //Find every countour within the image and test it to see if it is an eligible square (duplicates removed later)
    IsFindSquaresBusy=true; //set runtime boolean to busy (for receivepackets function)
    Squares.clear(); //clear the object/array which holds the vertices for each square so it is ready for a new iteration
    SquaresTempforMultiThreading.clear(); SquaresTempforMultiThreading.resize(NumberofThreads); //same idea as above -> clear all objects
    Mat pyr, gray0(AbsoluteOriginalImage.size(), CV_8U), gray; //Initialize Image Matrixes for various functions throughout
    if (TogglePyramidUpDownBlur) { //If blur is requested, carry out blurring procedures
    //The blurring procedure: down-scale and upscale the image to filter out the noise (another alternative is erode and dilate)
    pyrDown(AbsoluteOriginalImage, pyr, Size(AbsoluteOriginalImage.cols/2, AbsoluteOriginalImage.rows/2)); //downscale image
        pyrUp(pyr, gray0, AbsoluteOriginalImage.size()); //upscale image to original scaling
        if (DemoMode) {imshow(GUIWindowName, gray0); waitKey(DemoBeginFinishSleep/2);} //Display this image if in Demo Mode
    }
    else gray0=AbsoluteOriginalImage; //If blur is disabled, pass on original image as gray0 (gray original).
    if(DemoMode) {
        FindSquaresThreader(0, gray0, gray); //If in Demo Mode, run program with only one thread (otherwise it causes resource issues
        Squares=SquaresTempforMultiThreading[0]; //if in demo mode, only one array is used so go ahead and make the final array equal to the 0th array.
    }
    else { //otherwise, when not in demo mode, multithread the findsquares algorithm
    static thread Threads[NumberofThreads]; //Initialize Threading Objects
    for(int i=0; i<NumberofThreads; i++) {
        Threads[i]=thread(FindSquaresThreader, i, gray0, gray); //assign different starting indexes/points (variable i) to each thread
        }
    for(int i=0; i<NumberofThreads; i++) {
        Threads[i].join(); //wait until all threads finish execution before continuing
        }
    CombineThreadingData(); //Combine the squares each individual function found into one final squares object
    }
    IsFindSquaresBusy=false; //set runtime boolean to not busy (for receivepackets function)
}

void FindSquaresThreader(int startingindex, Mat gray0, Mat gray) { //this function is called multiple times as individual threads
    vector<vector<Point> > contours; //create a contours array/vector where all found contours are stored
    unsigned int ThreadNumber=NumberofThreads; // this and the next line checks whether the program is in demo mode or not and accordingly
    if (DemoMode) {ThreadNumber=1; startingindex=0;} //uses multiple threads or just one thread
    for( int l = startingindex; l < NumberofThresholdingIterations; l+=ThreadNumber) //Basically, in the end, this for loop is what is multithreaded
    {
        // Neat Feature: use the Canny edge detector instead of zero threshold level on the zeroth iteration
        // Canny helps to catch squares with gradient shading and makes the overall algorithm more robust
        if( l == 0  )
        {
            // apply Canny. Take the upper threshold from slider
            // and set the lower to 0 (which forces edges merging)
            Canny(gray0, gray, 0, CannyUpperLimitThreshold, 5);
            // dilate canny output to remove potential
            // holes between edge segments
            dilate(gray, gray, Mat(), Point(-1,-1));
        }
        else
        {
            // apply zero threshold if l!=0: (Read OpenCV Manual to learn the different types of thresholings available)
            //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
            gray = gray0 >= (l+1)*255/NumberofThresholdingIterations;
        }
        if(DemoMode) {imshow(GUIWindowName, gray); waitKey(DemoThresholdingSleep);} //you should know what this does by this point
        findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); // find contours and store them all as a list
        
        vector<Point> approx; //Temporary vector which only holds the vertice of one contour as it goes through the 'testing if it is a square' part of the algorithm
        
        // test each contour to see if it is an eligible square
        for( size_t i = 0; i < contours.size(); i++ )
        {
            // approximate contour with accuracy proportional to the contour perimeter
            // (You can choose this proportion as one of the starting setting variables passed from the Settings file
            approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*PolygonApproximationAccuracyProportion, true);
            
            // square contours should have 4 vertices after approximation
            // relatively large area (to filter out noisy contours) (I also included an upper limit as well)
            // and be convex.
            // Note: absolute value of an area is used because
            // area may be positive or negative - in accordance with the
            // contour orientation
            if( approx.size() == 4 &&
               fabs(contourArea(Mat(approx))) > MinAllowableSquareArea && fabs(contourArea(Mat(approx))) < MaxAllowableSquareArea &&
               isContourConvex(Mat(approx)) )
            {
                double maxCosine = 0;
                
                for( int j = 2; j < 5; j++ )
                {
                    // find the maximum cosine of the angle between joint edges
                    double cosine = fabs(MaxAngleFunction(approx[j%4], approx[j-2], approx[j-1]));
                    maxCosine = MAX(maxCosine, cosine);
                }
                
                // if cosines of all angles are small
                // (all angles are ~90 degree) then write quandrange
                // vertices to resultant sequence
                // at this point, if it passes all these requirements, it is an eligible square
                if( maxCosine < MaxAllowableAngle )
                    SquaresTempforMultiThreading[startingindex].push_back(approx); //write eligible squares to the temporary squares object
            } //note: notice how the object is first deferenced according to startingindex. This is becuase thread 0 will use the 0th array, thread 1 will use the 1st array, thread 2 will use the 2nd array, and etc.
        }
    }
    //if (DemoMode) {Squares=SquaresTempforMultiThreading[startingindex];} //if in demo mode, only one array is used so go ahead and make the final array equal to the 0th array. //moved to parent function
}

void CombineThreadingData () { //combine the squares from each thread into one object
    for (int i=0; i<SquaresTempforMultiThreading.size(); i++) {
        for (int j=0; j<SquaresTempforMultiThreading[i].size(); j++) {
            Squares.push_back(SquaresTempforMultiThreading[i][j]);
        }
    }
}

double MaxAngleFunction( Point pt1, Point pt2, Point pt0 ) {
    // helper function for the FindSquares Algorithm:
    // finds a cosine of angle between vectors
    // from pt0->pt1 and from pt0->pt2
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void RemoveIdenticalSquares () { //Does as it says -> it removes indentical squares. Also, it obtain the center of mass / center point for each square and stores it into a global variable called MassCenters. This data could be used for autoaim purposes, but we decided not to use it for the 2014 year season since vision was still in its debut mode. Definetly suggest using Center of Masses for later AutoAiming.
    if (Squares.size()!=0) {
        SquareInterMediate.clear();
        vector<Moments> massmoments(Squares.size() );
        double distance;
        for( int i = 0; i < Squares.size(); i++ )
        { massmoments[i] = moments( Squares[i], true ); } //tag each occurence of a square
        
        MassCenters.resize(Squares.size());
    
        for( int i = 0; i < Squares.size(); i++ ) ///  Get the mass centers:
        { MassCenters[i] = Point2f( massmoments[i].m10/massmoments[i].m00 , massmoments[i].m01/massmoments[i].m00 ); }
    
        for (int i1=0 ; i1<MassCenters.size()-1; i1++) {
            for (int i2=i1+1 ; i2<MassCenters.size(); i2++) {
                distance = sqrt(pow((MassCenters[i1].x-MassCenters[i2].x), 2) + pow((MassCenters[i1].y-MassCenters[i2].y), 2));
                if (distance<SquareCentersDistanceLimit) {MassCenters[i2]={0,0};} //this is where it uses the SquareCentersDistanceLimit to determine which squares should be considered duplicates
            }
        } //this and the next for loop is the part where it removes identical squares
        
        for (int i=0; i< Squares.size(); i++) {
            if (MassCenters[i].x!=0 && MassCenters[i].y!=0) {SquareInterMediate.push_back(Squares[i]); TemporaryCenters.push_back(MassCenters[i]);}
        }
        Squares.clear(); Squares=SquareInterMediate; MassCenters.clear(); MassCenters=TemporaryCenters; //rewrite squares and masscenters with the final remaining squares (No Duplicates)
        }
    else {MassCenters.resize(0);} //otherwise, if no squares were found in the first place, just resize the appropriate matrixes and move right along with the program
}

void UserInterfaces(const unsigned int iteration) { //I will leave this for you to decode and understand. This is where all User Interfaces are managed and outputed/etc. For GUI Mode and Demo Mode, this function also draws the final squares onto the original image before displaying that final image.
    if(textconsole) {if (Squares.size()==2) {cout<<"THIS IS A HOT GOAL\n ";} else {cout<<"Not a Hot Goal "<<Squares.size()<<"\n";}}
    if(GUIMODE || DemoMode || textconsole) {for( short i = 0; i < Squares.size(); i++ ) {
        if(textconsole) {cout<<Squares[i]<<"SQUARESIZE: "<<fabs(contourArea(Mat(Squares[i])))<<"\n";}
        if (GUIMODE || DemoMode) { //the draw squares part
            const Point* p = &Squares[i][0];
            int n = (int)Squares[i].size();
            polylines(AbsoluteOriginalImage, &p, &n, 1, true, Scalar(150,150,150), 3);}
    }}
    if(textconsole) cout<<"Total Square Area: "<<SquareArea()<<"\nFPS: "<<MilliSecondsPerRenderedFrames<<"\n\n\n";
    if(DemoMode || GUIMODE) {imshow(GUIWindowName, AbsoluteOriginalImage); displayOverlay(GUIWindowName,"Number of Squares: " + to_string(Squares.size())); if(DemoMode) {waitKey(DemoBeginFinishSleep); displayStatusBar(GUIWindowName,"Iteration Count: " + to_string(iteration));} if (GUIMODE) {displayStatusBar(GUIWindowName,"FPS: " + to_string(MilliSecondsPerRenderedFrames) + " | Iteration Count: " + to_string(iteration));} }
    waitKey(1);
    
}

void SendNetworkPackets(unsigned int i, unsigned int j, unsigned int k) { //sends desired data over UDP to the robot. I used it to send the # of squares and the ms required to render each frame. (It uses the time data to know when to read for UDP packets and hence the for loop iteration essentially stands for the # of frames it wants to read from the odroid.
    static QUdpSocket udpSocket; //create socket
    
    QByteArray datagram = QByteArray::number(i) + " " + QByteArray::number(j) + " " + QByteArray::number(k) + " "; //create data stream
    
    udpSocket.writeDatagram(datagram.data(), datagram.size(), QHostAddress(QstringTargetIPAddress), TargetPortNumber); //send that data stream over UDP to the TargetIpAddress with the given TargetPortNumber
}

void ReceiveNetworkPackets() { //waits for the LabView kill string, which indicates that LabView is done using the odroid and that the odroid should shutdown (managed in an upper level shell script which both executes this program and then shuts down the odroid after this program finishes/ends.
    const unsigned short datagramsize=LabViewKillString.size();
    QUdpSocket udpSocket; udpSocket.bind(HostPortNumber); //binds to HostPortNumber
    QByteArray datagram; datagram.resize(datagramsize);
    string str=" "; //Intermediate String for Later String Comparison
    while(!EndProgram) {
        if (!IsFindSquaresBusy) {
            udpSocket.readDatagram(datagram.data(), datagramsize);
            str=datagram.data();
            if(str==LabViewKillString) {EndProgram=true;}
            else usleep(ReceivePacketsDelay);
        }
        else usleep(FindSquares_Period); //So, basically, this thread will read UDP packets when FindSquares is not running and will do nothing except sleep when it is running to maximize the multithreading part of that algoritm. It waits as long as FindSquares takes to execute.
    }
    cout<<"DONE! Received LabView KillString! \n"; //Just a little dialog to know that it received the LabView killstring and that was why it shut down.
}

double SquareArea() { //returns the area of all final squares (We didn't really use or need this, but hey, here it is if you want it.
    double Area=0;
    if (Squares.size()==0) return 0;
    for (int i=0; i<Squares.size(); i++) {
        Area+=fabs(contourArea(Mat(Squares[i])));
    }
    return Area;
}

void DisplayGlobals() { //Self-Explanatory
    cout<<"\nNetworking is "<<(enablenetworking ? "ENABLED!" : "DISABLED!")<<"\n\n";
    cout<<"Target Ip Address= "<<TargetIPAddress<<'\n'<<
    "Target Port Number= "<<TargetPortNumber<<"\nHost Port Number= "<<HostPortNumber<<"\n\n"<<
    "The LabViewKillCode String is equal to: ||"<<LabViewKillString<<"||\nThe ReceivePacketDelayTime is "<<ReceivePacketsDelay<<" ms\n\n"<<
    "Text Output?= "<<truefalse(textconsole)<<'\n'<<
    "Final GUI Image?= "<<truefalse(GUIMODE)<<'\n'<<
    "Debugging Step-by-Step GUI? = "<<truefalse(DemoMode)<<" ;;; Processing SleepCount = "<<DemoThresholdingSleep<<" ;;; ProductSleepcount = "<<DemoBeginFinishSleep<<"\n\n";
    
    cout<<"PyramidUpDown Blur is "<<(TogglePyramidUpDownBlur ? "ON" : "OFF")<<"\n\n";
    
    cout<<"The Area Threshold is between +++ "<<MinAllowableSquareArea<<" +++ and +++ "<<MaxAllowableSquareArea<<"\n";
    
    cout<<"The Polygon Approximation Accuarcy Proportion is set to: "<<PolygonApproximationAccuracyProportion<<"\n";
    
    cout<<"The Angle Threshold is at +++ "<<MaxAllowableAngle<<"\n";
    
    cout<<"The Distance Threshold is at +++ "<<SquareCentersDistanceLimit<<"\n";
    
    cout<<"Canny Lower Limit Threshold is at +++ "<<CannyUpperLimitThreshold<<"\n";
    
    cout<<"The Number of Threshold Iterations is at +++ "<<NumberofThresholdingIterations<<"\n\n";
    
    cout<<"The Iteration Limit is set at +++ "<<ProgramIterationLimit<<"\n\n";
    
    cout<<"Number of Threads: "<<NumberofThreads<<"\n\n\n\n";
    
    if (ProgramIterationLimit>0) ProgramIterationLimit++; //an added thing to make sure it runs the correct number of iterations
}

void ReWriteGlobals(int argc, char** argv){ //Self-Explanatory
    if (argc==2) {
        if (file_exist_check(argv[1])==0) {
            cout<<"Error! The File Path You Provided Does NOT exist! -> Assuming Default Values\n\n";
        }
        else {
        cout<<"Settings File: \n"<<argv[1]<<"\n";
        string filepath = argv[1];
        ifstream settingsfile;
        settingsfile.open(filepath.c_str());
        settingsfile>>TargetIPAddress>>TargetPortNumber>>enablenetworking>>HostPortNumber>>LabViewKillString>>ReceivePacketsDelay>>textconsole>>GUIMODE>>DemoMode>>DemoThresholdingSleep>>DemoBeginFinishSleep>>TogglePyramidUpDownBlur>>MinAllowableSquareArea>>MaxAllowableSquareArea>>PolygonApproximationAccuracyProportion>>MaxAllowableAngle>>SquareCentersDistanceLimit>>CannyUpperLimitThreshold>>NumberofThresholdingIterations>>ProgramIterationLimit;
        settingsfile.close();
        QstringTargetIPAddress=QString::fromStdString(TargetIPAddress);
        if (DemoMode) {GUIMODE=false;}
        }
    }
    else {cout<<"Error! Path to Global Variables File was not provided or was provided with too many arguments -> Assuming Default Values\n\n";}
}

string truefalse (bool condition) {return condition ? "true": "false";} //You should be able to understand what this function does

inline bool file_exist_check (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
} //Checks whether a given file exists
//_____________________________________ (END OF PROGRAM SOURCE CODE)
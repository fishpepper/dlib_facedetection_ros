/******************************************************************************

Copyright 2016  Simon Schulz (University of Bielefeld)
                      [sschulz@techfak.uni-bielefeld.de]

********************************************************************************/

#ifndef INCLUDE_DLIB_FACEDETECTOR_NODELET_H_
#define INCLUDE_DLIB_FACEDETECTOR_NODELET_H_
#include <signal.h>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <string>
#include <vector>

#include "dlib_facedetector/dlib_facedetectorConfig.h"

namespace dlib_facedetector {

class Nodelet : public nodelet::Nodelet{
 public:
    Nodelet();
    ~Nodelet();

 private:
    void publishPeopleImage(const cv::Mat &image, std::vector<dlib::rectangle> faces,
                            std::vector<dlib::full_object_detection> shapes,
                            ros::Time timestamp);
    virtual void onInit();
    void imageCallback(const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&);
    void dynamicReconfigureCallback(const dlib_facedetector::dlib_facedetectorConfig &config,
                                    uint32_t level);
    void connectCb();

    volatile bool running_;
    boost::mutex connect_mutex_;
    dynamic_reconfigure::Server<dlib_facedetector::dlib_facedetectorConfig> *reconfig_server_;
    image_transport::ImageTransport *image_transport_;
    image_transport::CameraSubscriber image_subscriber_;

    ros::Publisher people_image_publisher_;
    ros::Publisher people_publisher_;

    std::string landmark_filename_;
    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;
    bool fit_landmarks_;
};

}  // namespace dlib_facedetector

#endif  // INCLUDE_DLIB_FACEDETECTOR_NODELET_H_


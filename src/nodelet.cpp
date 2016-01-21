/******************************************************************************

Copyright 2016  Simon Schulz (University of Bielefeld)
                      [sschulz@techfak.uni-bielefeld.de]

All rights reserved.

********************************************************************************/

#include "dlib_facedetector/nodelet.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <people_msgs/People.h>
#include <people_msgs/Person.h>
#include <string>
#include <vector>

// helper to convert dlib point to cv::Point
#define TO_CV_P(_p) cv::Point(_p.x(), _p.y())

using dlib_facedetector::Nodelet;
using std::vector;

Nodelet::Nodelet() : running_(false) {
    // nothing to do
    NODELET_INFO("Nodelet initialized");
}

Nodelet::~Nodelet() {
    image_subscriber_.shutdown();
    // saliency_image_publisher_.shutdown();
    // salient_spot_image_publisher_.shutdown();
    // salient_spot_publisher_.shutdown();
}

void Nodelet::imageCallback(const sensor_msgs::ImageConstPtr& image,
                            const sensor_msgs::CameraInfoConstPtr& info) {
    // extract image from msg:
    cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(image);

    // copy to dlib datatype
    dlib::cv_image<dlib::bgr_pixel> cimg(cv_image->image);

    // detect faces
    std::vector<dlib::rectangle> faces = detector(cimg);

    // find the pose of each face.
    std::vector<dlib::full_object_detection> shapes;
    if (fit_) {
        for (u_int32_t i = 0; i < faces.size(); ++i) {
            shapes.push_back(pose_model(cimg, faces[i]));
        }
    }

    // find persons and build people message:
    people_msgs::People people_msg;
    for (u_int32_t i = 0; i < faces.size(); ++i) {
        people_msgs::Person person_msg;
        person_msg.name = "unknown";
        person_msg.reliability = 0.0;

        double mid_x = (faces[i].left() + faces[i].right()) / 2.0;
        double mid_y = (faces[i].top() + faces[i].bottom()) / 2.0;

        person_msg.position.x = mid_x;
        person_msg.position.y = mid_y;
        person_msg.position.z = faces[i].right() - faces[i].left();

        people_msg.people.push_back(person_msg);
    }

    if (people_msg.people.size() > 0) {
        people_msg.header.stamp = image->header.stamp;
        people_publisher_.publish(people_msg);
    }

    // if requested publish debug image
    if (people_image_publisher_.getNumSubscribers()) {
        publishPeopleImage(cv_image->image, faces, shapes, image->header.stamp);
    }
}


void Nodelet::publishPeopleImage(const cv::Mat &image,
                                 std::vector<dlib::rectangle> faces,
                                 std::vector<dlib::full_object_detection> shapes,
                                 ros::Time timestamp) {
    // there is a subscriber, publish nice debug image
    cv::Mat debug_image = image.clone();

    for (u_int32_t i = 0; i < faces.size(); ++i) {
        // draw boxes
        cv::Rect rect(
                    faces[i].left(),
                    faces[i].top(),
                    faces[i].right() - faces[i].left(),
                    faces[i].bottom() - faces[i].top() );
        cv::rectangle(debug_image, rect, CV_RGB(255, 0, 0));

        // draw face markers
        if (fit_) {
            for (u_int32_t i = 0; i < shapes.size(); ++i) {
                // sanity check
                if (shapes[i].num_parts() != 68) {
                    NODELET_ERROR("ERROR: dlib invalid inputs given, expected 68 entries");
                    return;
                }

                cv::Scalar col = CV_RGB(0, 255, 0);

                const dlib::full_object_detection& d = shapes[i];

                for (u_int32_t i = 1; i <= 16; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);

                for (u_int32_t i = 28; i <= 30; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);

                for (u_int32_t i = 18; i <= 21; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);

                for (u_int32_t i = 23; i <= 26; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);

                for (u_int32_t i = 31; i <= 35; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);

                cv::line(debug_image, TO_CV_P(d.part(30)), TO_CV_P(d.part(35)), col);

                for (u_int32_t i = 37; i <= 41; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);
                cv::line(debug_image, TO_CV_P(d.part(36)), TO_CV_P(d.part(41)), col);

                for (u_int32_t i = 43; i <= 47; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);
                cv::line(debug_image, TO_CV_P(d.part(42)), TO_CV_P(d.part(47)), col);

                for (u_int32_t i = 49; i <= 59; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);
                cv::line(debug_image, TO_CV_P(d.part(48)), TO_CV_P(d.part(59)), col);

                for (u_int32_t i = 61; i <= 67; ++i)
                    cv::line(debug_image, TO_CV_P(d.part(i)), TO_CV_P(d.part(i-1)), col);
                cv::line(debug_image, TO_CV_P(d.part(60)), TO_CV_P(d.part(67)), col);
            }
        }
    }
}

// Handles (un)subscribing when clients (un)subscribe
void Nodelet::connectCb() {
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if ((people_publisher_.getNumSubscribers() == 0)
            && (people_image_publisher_.getNumSubscribers() == 0)) {
        NODELET_INFO("no more subscribers on people topics, shutting down image subscriber");
        image_subscriber_.shutdown();
    } else if (!image_subscriber_) {
        NODELET_INFO("new subscriber on people topic, subscribing to image topic");
        image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
        image_subscriber_ = image_transport_->subscribeCamera(
                    "image_color", 1, boost::bind(&Nodelet::imageCallback, this, _1, _2));
    }
}

void Nodelet::onInit() {
    NODELET_INFO("Nodelet::onInit()\n");

    ros::NodeHandle priv_nh(getPrivateNodeHandle());
    ros::NodeHandle node(getNodeHandle());

    if (!priv_nh.getParam("landmark_filename", landmark_filename_)) {
        NODELET_ERROR("missing parameter landmark_filename.  please pass face landmark"
                      "file for the shape predictor as parameter 'landmark_filename' "
                      "(e.g. in $prefix/share/dlib/shape_predictor_68_face_landmarks.dat)");
        exit(EXIT_FAILURE);
    }

    // load face db
    detector = dlib::get_frontal_face_detector();
    try {
        dlib::deserialize(landmark_filename_) >> pose_model;
    } catch (dlib::serialization_error &e) {
        NODELET_ERROR("ERROR: failed to load landmark file '%s'", landmark_filename_.c_str());
        NODELET_ERROR("You need dlib's default face landmarking model file to run this example.");
        NODELET_ERROR("You can get it from the following URL: ");
        NODELET_ERROR("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2");
        // NODELET_ERROR(std::string(e.what()));
        exit(EXIT_FAILURE);
    }

    image_transport_ = new image_transport::ImageTransport(node);

    // Monitor whether anyone is subscribed to the output
    image_transport::SubscriberStatusCallback imagetransport_connect_cb =
            boost::bind(&Nodelet::connectCb, this);

    ros::SubscriberStatusCallback connect_cb
            = boost::bind(&Nodelet::connectCb, this);

    // Make sure we don't enter connectCb() between advertising and assigning to pub_
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    people_image_publisher_ =
            image_transport_->advertise("/people/image_people", 1,
                                        imagetransport_connect_cb, imagetransport_connect_cb);

    people_publisher_ = node.advertise<people_msgs::People>("/detections", 10,
                                                            connect_cb, connect_cb);

    // attach to dyn reconfig server:
    NODELET_INFO("connecting to dynamic reconfiguration server");
    ros::NodeHandle reconf_node(priv_nh, "/saliency/parameters");
    reconfig_server_ =
        new dynamic_reconfigure::Server<dlib_facedetector::dlib_facedetectorConfig>(reconf_node);
    reconfig_server_->setCallback(boost::bind(&Nodelet::dynamicReconfigureCallback, this, _1, _2));

    running_ = true;
}


void Nodelet::dynamicReconfigureCallback(const dlib_facedetector::dlib_facedetectorConfig &config,
                                         uint32_t level) {
    fit_ = config.fit;
    /*boost::interprocess::scoped_lock<boost::mutex> lock(salient_spot_mutex_);
    salient_spot_ptr_.reset(new LQRPointTracker(
                                2,
                                config.dt,
                                config.drag,
                                config.move_cost));*/
}

// Register this plugin with pluginlib.  Names must match nodelet_velodyne.xml.
PLUGINLIB_EXPORT_CLASS(dlib_facedetector::Nodelet, nodelet::Nodelet);

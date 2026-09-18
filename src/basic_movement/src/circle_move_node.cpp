#include <cstdio>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp" // imports entire rclcpp API, narrow down later
#include "geometry_msgs/msg/twist.hpp" // The Twist message type expresses linear and angular velocity 
#include "geometry_msgs/msg/vector3.hpp"

// #include "nav_msgs/msg/odometry.hpp"

using namespace std::chrono_literals;

class CircleMoverNode : public rclcpp::Node{

    public:
        CircleMoverNode() : Node("circle_mover") { // need to actually give the Node constructor the right signature
            publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10); // 10 means limit backup queue to 10 messages, "KeepLast" scheme by detail (this is called QoS) 
            // subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>("/cmd_vel", 10, ...); 
            timer_ = this->create_wall_timer(500ms, std::bind(&CircleMoverNode::timer_velocity_callback, this));
            
        }

    private:
        // nav_msgs::msg::Odometry currentOdometry; // MARS publishes this to /odom topic
        geometry_msgs::msg::Twist nextVelocity; // MARS subscribes to /cmd_vel topic to get this. 
        
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;

        void timer_velocity_callback(){
            nextVelocity = geometry_msgs::msg::Twist();
            nextVelocity.linear.x = 1;
            nextVelocity.angular.z = 0.3;
            publisher_->publish(nextVelocity);
            RCLCPP_INFO(this->get_logger(), "Publishing Velocity with linear (1, 0, 0) and angular (0, 0, 0.3)");
        }

};

int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CircleMoverNode>());
    rclcpp::shutdown();
    return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <std_srvs/srv/trigger.hpp>
using namespace std::chrono_literals;
using std_srvs::srv::Trigger;

class ClientNode : public rclcpp::Node
{
  public:
    explicit ClientNode() : Node("client"){
      RCLCPP_INFO_STREAM(get_logger(), "client init");
      auto client_ = create_client<Trigger>("/trigger1");
      while (!client_->wait_for_service(2s)){
        RCLCPP_INFO_STREAM(get_logger(), "waiting for server...");
      };
      RCLCPP_INFO_STREAM(get_logger(), "found");
      auto req = std::make_shared<Trigger::Request>();
      auto future = client_->async_send_request(req);
      RCLCPP_INFO_STREAM(get_logger(), "trigger requested");
      rclcpp::spin_until_future_complete(this->shared_from_this(), future, 4s);
    }
  private:
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_;
};

int main(int argc, char * argv[])
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClientNode>());
  rclcpp::shutdown();
  return 0;
}

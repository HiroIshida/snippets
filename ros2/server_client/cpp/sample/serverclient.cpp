#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <std_srvs/srv/trigger.hpp>
using namespace std::chrono_literals;
using std_srvs::srv::Trigger;

class ServerClientNode : public rclcpp::Node
{
  public:
    ServerClientNode() : Node("serverclient"){
      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;

      client_ = create_client<Trigger>("/trigger2");
      while (!client_->wait_for_service(2s)){
        RCLCPP_INFO_STREAM(get_logger(), "waiting for server...");
      };
      RCLCPP_INFO_STREAM(get_logger(), "found server");
      srv_ = this->create_service<Trigger>(
        "/trigger1", std::bind(
          &ServerClientNode::onTrigger, this, _1, _2, _3));
    }
    bool onTrigger(
      const std::shared_ptr<rmw_request_id_t> request_header,
      const std::shared_ptr<Trigger::Request> request,
      std::shared_ptr<Trigger::Response> response){
      (void)request_header;
      (void)request;
      (void)response;
      RCLCPP_INFO_STREAM(get_logger(), "trigger 1 called");

      auto req = std::make_shared<Trigger::Request>();
      auto future = client_->async_send_request(req);
      RCLCPP_INFO_STREAM(get_logger(), "trigger 2 requested");
      rclcpp::spin_until_future_complete(this->shared_from_this(), future, 4s);

      return true;
    };
  private:
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_;
};

int main(int argc, char * argv[])
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ServerClientNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

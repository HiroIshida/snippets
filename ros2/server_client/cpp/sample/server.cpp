#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <std_srvs/srv/trigger.hpp>
using namespace std::chrono_literals;
using std_srvs::srv::Trigger;

class ServerNode : public rclcpp::Node
{
  public:
    ServerNode() : Node("server"){
      RCLCPP_INFO_STREAM(get_logger(), "server init");
      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;

      srv_ = this->create_service<Trigger>(
        "/trigger2", std::bind(
          &ServerNode::onTrigger, this, _1, _2, _3));
    }
    bool onTrigger(
      const std::shared_ptr<rmw_request_id_t> request_header,
      const std::shared_ptr<Trigger::Request> request,
      std::shared_ptr<Trigger::Response> response){
      (void)request_header;
      (void)request;
      (void)response;
      RCLCPP_INFO_STREAM(get_logger(), "trigger 2 called");
      return true;
    };
  private:
    rclcpp::Service<Trigger>::SharedPtr srv_;
};

int main(int argc, char * argv[])
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ServerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

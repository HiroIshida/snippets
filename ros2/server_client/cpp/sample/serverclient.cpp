#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <std_srvs/srv/trigger.hpp>
using namespace std::chrono_literals;
using std_srvs::srv::Trigger;

class ServerClientNode : public rclcpp::Node
{
  public:
    ServerClientNode() : Node("serverclient"){
      RCLCPP_INFO_STREAM(get_logger(), "serverclient init");
      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;

      callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      client_ = this->create_client<Trigger>("/trigger2", rmw_qos_profile_services_default, callback_group_);

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

      const auto inner_callback = [&](rclcpp::Client<Trigger>::SharedFuture inner_future){
          auto result = inner_future.get();
          RCLCPP_INFO(this->get_logger(), "[inner service] callback executed");
          RCLCPP_INFO_STREAM(this->get_logger(), result->message);
      };
      auto req = std::make_shared<Trigger::Request>();
      auto future = client_->async_send_request(req, inner_callback);
      RCLCPP_INFO_STREAM(get_logger(), "trigger 2 requested");

      //rclcpp::spin_until_future_complete(this->shared_from_this(), future, 4s);
      std::future_status status = future.wait_for(10s);
      RCLCPP_INFO_STREAM(get_logger(), "completed");

      return true;
    };
  private:
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;
};

int main(int argc, char * argv[])
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<ServerClientNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}

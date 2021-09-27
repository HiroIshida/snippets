int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<Communication>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

hoge(){
callback_group_input_ = this->create_callback_group(rclcpp::callback_group::CallbackGroupType::MutuallyExclusive);
get_input_client_ = this->create_client<petra_core::srv::GetInput>("GetInput", rmw_qos_profile_services_default, callback_group_input_);

int choice = -1;
    auto inner_client_callback = [&,this](rclcpp::Client<petra_core::srv::GetInput>::SharedFuture inner_future)
        { 
            auto result = inner_future.get();
            choice = stoi(result->input);
            RCLCPP_INFO(this->get_logger(), "[inner service] callback executed");
        };
    auto inner_future_result = get_input_client_->async_send_request(inner_request, inner_client_callback);
}

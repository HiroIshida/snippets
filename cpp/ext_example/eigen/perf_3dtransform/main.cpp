#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include "urdf_pose.hpp"

template<typename Scalar>
struct MatAndTrans {
    Eigen::Matrix<Scalar, 3, 3> R;
    Eigen::Matrix<Scalar, 3, 1> t;
};

template<typename Scalar>
struct QuatAndTrans {
    Eigen::Quaternion<Scalar> q;
    Eigen::Matrix<Scalar, 3, 1> t;
};

template<typename Scalar>
using Transform = Eigen::Transform<Scalar, 3, Eigen::Affine>;

template<typename Scalar>
Transform<Scalar> matAndTransToTransform(const MatAndTrans<Scalar>& mat) {
    Transform<Scalar> transform = Transform<Scalar>::Identity();
    transform.linear() = mat.R;
    transform.translation() = mat.t;
    return transform;
}

template<typename Scalar>
Transform<Scalar> quatAndTransToTransform(const QuatAndTrans<Scalar>& quat) {
    Transform<Scalar> transform = Transform<Scalar>::Identity();
    transform.linear() = quat.q.toRotationMatrix();
    transform.translation() = quat.t;
    return transform;
}

template<typename Scalar>
MatAndTrans<Scalar> transformToMatAndTrans(const Transform<Scalar>& transform) {
    MatAndTrans<Scalar> result;
    result.R = transform.linear();
    result.t = transform.translation();
    return result;
}

template<typename Scalar>
QuatAndTrans<Scalar> transformToQuatAndTrans(const Transform<Scalar>& transform) {
    QuatAndTrans<Scalar> result;
    result.q = Eigen::Quaternion<Scalar>(transform.linear());
    result.t = transform.translation();
    return result;
}

urdf::Pose transformToUrdfPose(const Transform<double>& transform) {
    urdf::Pose pose;
    pose.position.x = transform.translation().x();
    pose.position.y = transform.translation().y();
    pose.position.z = transform.translation().z();
    Eigen::Quaterniond q(transform.linear());
    pose.rotation.x = q.x();
    pose.rotation.y = q.y();
    pose.rotation.z = q.z();
    pose.rotation.w = q.w();
    return pose;
}

template<typename Scalar>
Transform<Scalar> randomTransform() {
    Eigen::Matrix<Scalar, 3, 1> axis = Eigen::Matrix<Scalar, 3, 1>::Random().normalized();
    Scalar angle = static_cast<Scalar>(rand()) / RAND_MAX * 2 * M_PI;
    Eigen::Matrix<Scalar, 3, 1> translation = Eigen::Matrix<Scalar, 3, 1>::Random() * 10;
    Transform<Scalar> t = Transform<Scalar>::Identity();
    t.linear() = Eigen::AngleAxis<Scalar>(angle, axis).toRotationMatrix();
    t.translation() = translation;
    return t;
}


template<typename Scalar>
void runBenchmark() {
    std::cout << "Running benchmark with " << (std::is_same<Scalar, float>::value ? "float" : "double") << std::endl;
    
    size_t N = 100000000;
    auto tf1_original = randomTransform<Scalar>();
    auto tf1 = tf1_original;
    auto tf2 = randomTransform<Scalar>();

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            tf1 = tf1 * tf2;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench Affine3d: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << tf1.translation().transpose() << std::endl; // to avoid compiler optimization
    }

    {
        auto matatrans1 = transformToMatAndTrans(tf1_original);
        auto matatrans2 = transformToMatAndTrans(tf2);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            matatrans1.R = matatrans1.R * matatrans2.R;
            matatrans1.t = matatrans1.R * matatrans2.t + matatrans1.t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench MatAndTrans: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << matatrans1.t.transpose() << std::endl; // to avoid compiler optimization
    }

    {
        auto quatatrans1 = transformToQuatAndTrans(tf1_original);
        auto quatatrans2 = transformToQuatAndTrans(tf2);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            quatatrans1.q = quatatrans1.q * quatatrans2.q;
            quatatrans1.t = quatatrans1.q * quatatrans2.t + quatatrans1.t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench QuatAndTrans: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << quatatrans1.t.transpose() << std::endl; // to avoid compiler optimization
    }
}

void runBenchmark_urdf_pose() {
    std::cout << "Running benchmark with UrdfPose" << std::endl;
    size_t N = 100000000;
    auto tf1_original = randomTransform<double>();
    auto tf1 = tf1_original;
    auto tf2 = randomTransform<double>();
    auto urdfpose1 = transformToUrdfPose(tf1_original);
    auto urdfpose2 = transformToUrdfPose(tf2);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
      urdfpose1 = urdf::pose_transform(urdfpose1, urdfpose2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bench UrdfPose: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << urdfpose1.position.x << " " << urdfpose1.position.y << " " << urdfpose1.position.z << std::endl; // to avoid compiler optimization
}

int main() {
    runBenchmark<float>();
    runBenchmark<double>();
    runBenchmark_urdf_pose();
    return 0;
}

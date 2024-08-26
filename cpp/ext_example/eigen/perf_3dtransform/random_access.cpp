#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <vector>
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
struct Homogeneous {
    Eigen::Matrix<Scalar, 4, 4> H;
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

template<typename Scalar>
Homogeneous<Scalar> transformToHomogeneous(const Transform<Scalar>& transform) {
    Eigen::Matrix< Scalar, 4, 4 > mat = Eigen::Matrix< Scalar, 4, 4 >::Identity();
    mat.block(0, 0, 3, 3) = transform.linear();
    mat.block(0, 3, 3, 1) = transform.translation();
    return {mat};
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

std::vector<size_t> generateRandomIndices(size_t count, size_t max_index) {
    std::vector<size_t> indices(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_index - 1);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = dis(gen);
    }
    return indices;
}

template<typename Scalar>
void runBenchmark() {
    std::cout << "Running benchmark with " << (std::is_same<Scalar, float>::value ? "float" : "double") << std::endl;
    
    const size_t N = 10000000;  // bench number
    const size_t M = 200;  // number of random transformations
    std::vector<Transform<Scalar>> transforms(M);
    std::vector<MatAndTrans<Scalar>> matAndTransforms(M);
    std::vector<QuatAndTrans<Scalar>> quatAndTransforms(M);
    std::vector<Homogeneous<Scalar>> homogeneousTransforms(M);

    for (size_t i = 0; i < M; ++i) {
        transforms[i] = randomTransform<Scalar>();
        matAndTransforms[i] = transformToMatAndTrans(transforms[i]);
        quatAndTransforms[i] = transformToQuatAndTrans(transforms[i]);
        homogeneousTransforms[i] = transformToHomogeneous(transforms[i]);
    }

    std::vector<size_t> randomIndices = generateRandomIndices(N, M);

    {
        auto result = Transform<Scalar>::Identity();
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            result = result * transforms[randomIndices[i]];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench Affine3d: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << result.translation().transpose() << std::endl; // to avoid compiler optimization
    }

    {
        MatAndTrans<Scalar> result = {Eigen::Matrix<Scalar, 3, 3>::Identity(), Eigen::Matrix<Scalar, 3, 1>::Zero()};
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            const auto& mat = matAndTransforms[randomIndices[i]];
            result.R = result.R * mat.R;
            result.t = result.R * mat.t + result.t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench MatAndTrans: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << result.t.transpose() << std::endl; // to avoid compiler optimization
    }

    {
        QuatAndTrans<Scalar> result = {Eigen::Quaternion<Scalar>::Identity(), Eigen::Matrix<Scalar, 3, 1>::Zero()};
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            const auto& quat = quatAndTransforms[randomIndices[i]];
            result.q = result.q * quat.q;
            result.t = result.q * quat.t + result.t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench QuatAndTrans: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << result.t.transpose() << std::endl; // to avoid compiler optimization
    }

    {
        Homogeneous<Scalar> result = {Eigen::Matrix<Scalar, 4, 4>::Identity()};
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; ++i) {
            result.H = result.H * homogeneousTransforms[randomIndices[i]].H;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bench Homogeneous: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << result.H(0, 3) << " " << result.H(1, 3) << " " << result.H(2, 3) << std::endl; // to avoid compiler optimization
    }
}

void runBenchmark_urdf_pose() {
    std::cout << "Running benchmark with UrdfPose" << std::endl;
    const size_t N = 10000000;  // bench number
    const size_t M = 200;  // number of random transformations
    std::vector<urdf::Pose> urdfPoses(N);

    for (size_t i = 0; i < M; ++i) {
        urdfPoses[i] = transformToUrdfPose(randomTransform<double>());
    }

    std::vector<size_t> randomIndices = generateRandomIndices(N, M);

    urdf::Pose result;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        result = urdf::pose_transform(result, urdfPoses[randomIndices[i]]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bench UrdfPose: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << result.position.x << " " << result.position.y << " " << result.position.z << std::endl; // to avoid compiler optimization
}

int main() {
    // runBenchmark<float>();
    runBenchmark<double>();
    runBenchmark_urdf_pose();
    return 0;
}

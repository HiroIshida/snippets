#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>

#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/jntarray.hpp>
#include <urdf/model.h>

#define HDR1 0xAA
#define HDR2 0x55
#define DOF 7
#define ARRAY_SIZE 14
#define RESPONSE_ARRAY_SIZE 21


class DynamicsParameterCalculator {
public:
  DynamicsParameterCalculator() {
    std::string urdf_path = "/home/h-ishida/ros2_ws/src/openarm_ros2/openarm_bimanual_description/urdf/openarm_bimanual.urdf";
    std::string chain_root_link = "pedestal_link";
    std::string left_leaf_link = "left_link8";

    urdf::Model urdf_model;

    if (!urdf_model.initFile(urdf_path)) {
      throw std::runtime_error("Failed to parse URDF");
    }
    if (!kdl_parser::treeFromUrdfModel(urdf_model, kdl_tree_)) {
      throw std::runtime_error("Failed to parse KDL tree");
    }
    if (!kdl_tree_.getChain(chain_root_link, left_leaf_link, kdl_chain_)) {
      throw std::runtime_error("Failed to get KDL chain");
    }
    if(kdl_chain_.getNrOfJoints() != DOF) {
      throw std::runtime_error("DOF mismatch");
    }

    solver_ = std::make_unique<KDL::ChainDynParam>(kdl_chain_, KDL::Vector(0, 0, -9.81));

    // dynamic allocation first
    q_.resize(DOF +1);
    q_dot_.resize(DOF);
    gravity_.resize(DOF);
    coriolis_.resize(DOF);
    M_.resize(DOF);
  }

  void compute_params(const std::array<float, ARRAY_SIZE>& inp, std::array<float, RESPONSE_ARRAY_SIZE>& out) {
    // internally cast from float to double:
    for(size_t i = 0; i < DOF; ++i) {
      q_(i) = inp.at(i);
      q_dot_(i) = inp.at(i + DOF);
    }

    solver_->JntToGravity(q_, gravity_);
    solver_->JntToCoriolis(q_, q_dot_, coriolis_);
    solver_->JntToMass(q_, M_);

    // pack (M_diag, coriolis, gravity)
    for(size_t i = 0; i < DOF; ++i) {
      out[i] = M_(i, i);
      out[i + DOF] = coriolis_(i);
      out[i + 2 * DOF] = gravity_(i);
    }
  }


private:
  KDL::Chain kdl_chain_;
  KDL::Tree kdl_tree_;
  std::unique_ptr<KDL::ChainDynParam> solver_;
  KDL::JntArray q_;
  KDL::JntArray q_dot_;
  KDL::JntArray gravity_;
  KDL::JntArray coriolis_;
  KDL::JntSpaceInertiaMatrix M_;
};


bool configureSerial(int fd, speed_t baudrate) {
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        std::perror("tcgetattr");
        return false;
    }

    cfmakeraw(&tty);
    cfsetispeed(&tty, baudrate);
    cfsetospeed(&tty, baudrate);

    tty.c_cflag |= CLOCAL | CREAD;    // ローカル接続 & 受信有効
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;               // 8ビット
    tty.c_cflag &= ~PARENB;           // パリティ無し
    tty.c_cflag &= ~CSTOPB;           // ストップビット1

    tty.c_cc[VMIN]  = 1;    // 最低1バイト受信
    tty.c_cc[VTIME] = 10;   // タイムアウト 1.0秒

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::perror("tcsetattr");
        return false;
    }
    return true;
}

// 指定バッファからバイト単位のCRC(XOR)を計算
uint8_t calcCRC(const uint8_t* data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
    }
    return crc;
}

// ヘッダ検出：0xAA 0x55 の順に読めるまでループ
bool waitForHeader(int fd) {
    uint8_t byte;
    // 最初のヘッダバイトを探す
    while (true) {
        ssize_t n = read(fd, &byte, 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::perror("read");
            return false;
        }
        if (n == 0) continue;
        if (byte == HDR1) break;
    }
    // 次に 0x55 をチェック
    while (true) {
        ssize_t n = read(fd, &byte, 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::perror("read");
            return false;
        }
        if (n == 0) continue;
        if (byte == HDR2) return true;
        // 違うなら最初のヘッダに戻る
        if (byte == HDR1) continue;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <serial_device>\n";
        return EXIT_FAILURE;
    }
    const char* device = argv[1];

    int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::perror("open");
        return EXIT_FAILURE;
    }
    if (!configureSerial(fd, B115200)) {
        close(fd);
        return EXIT_FAILURE;
    }

    const size_t payloadBytes = ARRAY_SIZE * sizeof(float);
    const size_t responseBytes = RESPONSE_ARRAY_SIZE * sizeof(float);

    std::vector<uint8_t> buf(payloadBytes);
    std::array<float, ARRAY_SIZE> data_req;
    std::array<float, RESPONSE_ARRAY_SIZE> data_resp;
    std::vector<uint8_t> outbuf(responseBytes);

    auto dynparam_calculator = DynamicsParameterCalculator();

    while (true) {
        // 1) ヘッダ受信
        if (!waitForHeader(fd)) {
            std::cerr << "Header error\n";
            break;
        }

        // 2) データ本体読み込み
        size_t readTotal = 0;
        while (readTotal < payloadBytes) {
            ssize_t n = read(fd, buf.data() + readTotal, payloadBytes - readTotal);
            if (n < 0) {
                if (errno == EINTR) continue;
                std::perror("read payload");
                break;
            }
            readTotal += n;
        }
        if (readTotal < payloadBytes) break;

        // 3) CRC読み込み
        uint8_t recvCrc;
        if (read(fd, &recvCrc, 1) != 1) {
            std::perror("read crc");
            break;
        }

        // 4) CRC検証
        uint8_t calc_crc = calcCRC(buf.data(), payloadBytes);
        if (calc_crc != recvCrc) {
            std::cerr << "CRC mismatch: calc=" << int(calc_crc)
                      << " recv=" << int(recvCrc) << "\n";
            continue;
        }

        // 5) compute
        std::memcpy(data_req.data(), buf.data(), payloadBytes);
        dynparam_calculator.compute_params(data_req, data_resp);

        // convert response to uint8_t
        std::memcpy(outbuf.data(), data_resp.data(), responseBytes);
        uint8_t outCrc = calcCRC(outbuf.data(), responseBytes);

        // 6) 返信送信（ヘッダ＋データ＋CRC）
        uint8_t hdr[2] = { HDR1, HDR2 };
        if (write(fd, hdr, 2) != 2 ||
            write(fd, outbuf.data(), responseBytes) != (ssize_t)responseBytes ||
            write(fd, &outCrc, 1) != 1) {
            std::perror("write reply");
            break;
        }

        std::cout << "Replied +1 to all floats\n";
    }

    close(fd);
    return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

#define HDR1 0xAA
#define HDR2 0x55
#define ARRAY_SIZE 14

int configureSerial(int fd, speed_t baudrate) {
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return 0;
    }

    cfmakeraw(&tty);
    cfsetispeed(&tty, baudrate);
    cfsetospeed(&tty, baudrate);

    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;

    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 10; // timeout in deciseconds

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return 0;
    }

    return 1;
}

void sendFloatArray(int fd, float arr[ARRAY_SIZE]) {
    uint8_t crc = 0;
    uint8_t header[2] = { HDR1, HDR2 };
    write(fd, header, sizeof(header));

    const uint8_t* p = (const uint8_t*)arr;
    size_t bytes = ARRAY_SIZE * sizeof(float);

    for (size_t i = 0; i < bytes; ++i) {
        uint8_t b = p[i];
        write(fd, &b, 1);
        crc ^= b;
    }

    write(fd, &crc, 1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <serial_device>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* device = argv[1];
    int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    if (!configureSerial(fd, B115200)) {
        close(fd);
        return EXIT_FAILURE;
    }

    float data[ARRAY_SIZE] = { // 7 for q and 7 for v
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f
    };

    while(1) {
        printf("Sent data\n");
        sendFloatArray(fd, data);
        usleep(50000);
    }

    close(fd);
    return EXIT_SUCCESS;
}

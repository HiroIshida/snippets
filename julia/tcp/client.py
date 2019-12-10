import socket

target_ip = "127.0.0.1"
target_port = 2000
buffer_size = 4096

tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_client.connect((target_ip,target_port))
tcp_client.send(b"Data by TCP Client!!")
response = tcp_client.recv(buffer_size)
print(response)
#print("[*]Received a response : {}".format(response))

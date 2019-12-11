import socket

HOST = "127.0.0.1"
PORT = 2000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

# these two steps are required because julia's readline expecet "\n" fast
s.send("\n")
response = s.recv(1024)

def g(str):
    s.send(str+"\n")
    response = s.recv(1024)
    print(response)


f = gen()



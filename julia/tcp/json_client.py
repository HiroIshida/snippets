import socket
import json

HOST = "127.0.0.1"
PORT = 2000

dict1 = {"method_name": "method1", 
        "args": (2)}
dict2 = {"method_name": "method2", 
        "args": (2)}
dict3 = {"method_name": "method3", 
        "args": (2, 3)}
j1, j2, j3 = [json.dumps(d) for d in [dict1, dict2, dict3]]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

# these two steps are required because julia's readline expecet "\n" fast
s.send("\n")
response = s.recv(1024)

def g(str):
    s.send(str+"\n")
    response = s.recv(1024)
    print(response)



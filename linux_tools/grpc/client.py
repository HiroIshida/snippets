import numpy as np
import pickle
import grpc
import datagen_pb2
import datagen_pb2_grpc


with grpc.insecure_channel('localhost:5051') as channel:
    stub = datagen_pb2_grpc.DataGenServiceStub(channel)
    data = pickle.dumps(np.random.randn(100))
    req = datagen_pb2.DataGenRequest(data=data)
    response = stub.DataGen(req)

print('Reply: ', response.data)

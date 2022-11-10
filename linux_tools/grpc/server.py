import time
import grpc
import datagen_pb2_grpc
import datagen_pb2
from concurrent import futures


class Server(datagen_pb2_grpc.DataGenServiceServicer):
    # echo server

    def DataGen(self, request: datagen_pb2.DataGenRequest, context) -> datagen_pb2.DataGenResponse:
        req_data: bytes = request.data
        response = datagen_pb2.DataGenResponse(data = req_data)
        return response

    def DataGenStream(self, request, context):
        req_data: bytes = request.data
        for _ in range(10):
            time.sleep(1)
            response = datagen_pb2.DataGenResponse(data = req_data)
            yield response


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
datagen_pb2_grpc.add_DataGenServiceServicer_to_server(Server(), server)
server.add_insecure_port('[::]:5051')
server.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop(0)

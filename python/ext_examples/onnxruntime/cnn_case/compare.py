import argparse
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
import onnxruntime as ort

def define_encoder():
    encoder_layers = [
        nn.Conv2d(1, 8, 3, padding=1, stride=(2, 2)),  # 14x14
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(1024, 1000),
        nn.ReLU(inplace=True),
    ]
    encoder = nn.Sequential(*encoder_layers)
    return encoder

if __name__ == "__main__":
    # get args determine whether to use GPU or CPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_onnxruntime', action='store_true')
    args = parser.parse_args()

    encoder = define_encoder()
    x = torch.zeros(1, 1, 56, 56)

    N = 100 
    if not args.use_onnxruntime:
        device = torch.device("cuda" if args.use_gpu else "cpu")
        x = x.to(device)
        encoder = encoder.to(device)
        start = time.time()
        for _ in range(N):
            y = encoder(x)
        end = time.time()
        print(f"torch with {device}: ", (end - start) / N)
    else:
        # bench onnxruntime
        onnx_file = "encoder.onnx"
        torch.onnx.export(encoder, x, onnx_file, 
                          export_params=True,   # store the trained parameter weights inside the model file
                          opset_version=11,     # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output']) # the model's output names
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'], sess_options=so)

        x_numpy = x.cpu().numpy()
        start = time.time()
        for _ in range(N):
            ort_outputs = ort_session.run(None, {'input': x_numpy})
        end = time.time()
        print("ONNX Runtime: ", (end - start) / N)

import torch
from model import SqueezeNet
import torch.onnx

device = torch.device('cpu') #"cuda:0" if torch.cuda.is_available() else "cpu")

model = SqueezeNet(42).to(device)
model.load_state_dict(torch.load('saves/squeezenet_42c_86acc'))
model.eval()

batch_size = 1

x = torch.randn(batch_size, 1, 127, 127, requires_grad=True).to(device)
torch_out = model(x)

torch.onnx.export(model, x, 'saves/squeezenet.onnx',
    input_names=['input'], output_names=['output'],
    do_constant_folding=True,
    dynamic_axes={
        'input': {
            0: 'batch_size',
            2: 'height',
            3: 'width'
        }
    }
)

import onnx

onnx_model = onnx.load("squeezenet.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("saves/squeezenet.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
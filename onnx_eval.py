import sys
from PIL import Image
from PIL.ImageOps import autocontrast
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("saves/squeezenet.onnx")

def transform(image):
    # Grayscale
    image = image.convert('L')
    # Autocontrast
    image = autocontrast(image)
    # Resize
    size = 127
    w, h = image.size
    oh = size
    ow = int(size * w / h)
    image = image.resize((ow, oh), Image.BILINEAR)
    # ToTensor
    pic = np.array(image)
    pic = pic[:, :, None]
    pic = pic.transpose((2, 0, 1))
    pic = pic.astype(np.float32, copy=False) / 255.0
    # unsqueeze
    pic = np.expand_dims(pic, axis=0)

    return pic

def classify(image):
    inputs = transform(image)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return np.argsort(-ort_outs[0][0])[:5].tolist()

if __name__ == '__main__':
    print(classify(Image.open(sys.argv[1])))
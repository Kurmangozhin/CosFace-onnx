import onnxruntime as ort
import cv2
import numpy as np


def cosine_dist_python(A,B):
    A_dot_B = np.dot(A,B)
    A_mag = np.sqrt(np.sum(np.square(A)))
    B_mag = np.sqrt(np.sum(np.square(B)))
    dist = 1.0 - (A_dot_B / (A_mag * B_mag))
    return dist


class CosFace(object):
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def inference_trt(self, image_1):
        ort_inputs = {self.sess.get_inputs()[0].name: image_1[None, :, :, :]}
        ort_inputs = self.sess.run(None, ort_inputs)[0]
        return ort_inputs

    @staticmethod
    def rescale(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = image / 255.
        return image


    def proccessing(self, image_path: str) -> np.ndarray:
        image_1 =cv2.imread(image_path)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_1 = cv2.resize(image_1, (112, 112))
        image_1 = self.rescale(image_1)
        image_1 = np.transpose(image_1, (2, 0, 1)).astype(np.float32)
        return image_1

    def __call__(self, image_path: str):
        imgz = self.proccessing(image_path)
        features = self.inference_trt(imgz)[0]
        return features



if __name__ == '__main__':
    module = CosFace("models/sphere20.onnx")
    emb1 = module("...jpg") # image_path_1
    emb2 = module("....jpg") # image_path_2
    dist = cosine_dist_python(emb1, emb2)
    print(dist)
    

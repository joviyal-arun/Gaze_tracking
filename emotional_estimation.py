import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os
import ngraph as ng

class EmotionalEstimation:
    '''
    Class for the Emotional Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        model_xml, model_bin = str(model_name), str(os.path.splitext(model_name)[0] + ".bin")
        self.core = IECore()
        self.device=device
        self.emotional = self.core.read_network(model=model_xml, weights = model_bin)
##        self.input_blob = next(iter(self.emotional.inputs)) # Older One
        self.input_blob = next(iter(self.emotional.input_info))
        self.out_blob = next(iter(self.emotional.outputs))

    def load_model(self):
        self.exec_net = self.core.load_network(network=self.emotional, device_name="CPU")
        return self.exec_net

    def sync_inference(self, image):
        input_blob = next(iter(self.exec_net.input_info))
        return self.exec_net.infer({input_blob: image})
        
    def async_inference(self, image, request_id=0):
        # create async network
        input_blob = next(iter(self.exec_net.inputs))
        async_net = self.exec_net.start_async(request_id, inputs={input_blob: image})

        # perform async inference
        output_blob = next(iter(async_net.outputs))
        status = async_net.requests[request_id].wait(-1)
        if status == 0:
            result = async_net.requests[request_id].outputs[output_blob]
        return result


    def check_model(self):
        
        if "CPU" in self.device:

            ngraph_func = ng.function_from_cnn(self.emotional)
                
            supported_layers = self.core.query_network(network=self.emotional, device_name=self.device)

            unsupported_layers = [l for l in ngraph_func.get_ordered_ops() if l.get_friendly_name() not in supported_layers]

            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)

    def preprocess_input(self, image):

        n, c, h, w = self.emotional.input_info[self.input_blob].tensor_desc.dims
        image = cv2.resize(image, (w, h))
        image = image.transpose(2,0,1)
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        
        emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger'] 

        emotional   = outputs['prob_emotion']
        emotional = np.argmax(emotional) 
        emotional = emotions[emotional]

        probaibility=outputs['prob_emotion']
        index=np.argmax(probaibility) 
        probaibility_score=probaibility[0][index][0][0]*100

        
        return emotional,probaibility_score


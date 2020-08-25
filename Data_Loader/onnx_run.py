import numpy as np
import onnxruntime as rt
#import numpy as np
#from PIL import Image, ImageFont, ImageDraw
import time
import cv2
import os
import colorsys
import imageio
from PIL import Image



img_name="4.jpg"

def reshape_image(image):
    image = np.asarray(imageio.imread(image))
    h, w, _ = image.shape
    image = Image.fromarray(image)
    image = image.resize((100, 100))
    image = np.asarray(image)
    # import cv2
    # cv2.imshow("", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = np.transpose(image, (2, 0, 1))
    image = np.reshape(image, (1, 3, 100, 100)).astype(np.float32)
    print(image.shape)
    return image

image_data=reshape_image(img_name)

model_path="torch_model.onnx"
#model_session = rt.InferenceSession(model_path)
#input_name1 = model_session.get_inputs()[0].name
#input_name2 = model_session.get_inputs()[1].name
#input_name=[inputx.name for inputx in model_session.get_inputs()]
#model_session.run(outname,input_name)

#print(input_name)
#print(outname)

model_session = rt.InferenceSession(model_path)
input_name1 = model_session.get_inputs()[0].name
#input_name2 = model_session.get_inputs()[1].name

outname = [output.name for output in model_session.get_outputs()]
print("Inputs: {}".format("\n".join([str(i) for i in model_session.get_inputs()])))
print("Outputs: {}".format("\n".join([str(i) for i in model_session.get_outputs()])))

#data=[image_data,image_size]
#print(input_name1,input_name2)
#inp_dim = model_session.get_inputs()[0].shape[2]
##print(inp_dim)
#dat = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)



#outname = [output.name for output in model_session.get_outputs()]


#dict_input = {input_name1 : image_data,
#              input_name2: image_size }
#output_name = model_session.get_outputs()[1]
#print(outname)


results = model_session.run(outname,{input_name1:image_data})
print(len(results))
for i in results:
    print(i)

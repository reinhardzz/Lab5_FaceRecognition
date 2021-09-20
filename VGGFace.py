#Face encoding: VGGFace  

from matplotlib import pyplot
from numpy import asarray
import cv2

model_VGG = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
faces_encode = []
for face in faces:
  sample = cv2.resize(face, (224,224))

  samples = list([sample])
  samples = asarray(samples, 'float32')

  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=2)
  # create a vggface model
  # perform encoding
  yhat = model_VGG.predict(samples)
  faces_encode.append(yhat)
  pyplot.plot(range(len(yhat.T)),yhat.T)
#faces_encode.append(yhat)

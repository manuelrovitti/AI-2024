import tensorflow as tf
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np
import os


def get_prob_array(model, data):
  prob_array=[]
  prov_array=[]
  for one_case in range(0,len(data)):
    predictions_single = model.predict(data[one_case:(one_case + 1)])
    prob_array.append(predictions_single)
    prov_array.append(predictions_single.argmax())
  return prob_array, prov_array

def predict(f_input, bgty_model):
	#carico le celle della griglia sul modello
	i,width,height=0,0,0
	data=[]
	folder = os.listdir(f_input)
	folder = [os.path.splitext(x)[0] for x in folder]
	folder.sort(key=lambda x: int(x))

	height = max([int(x[:1]) for x in folder]) + 1
	width = max([int(x[1:]) for x in folder]) + 1

	for filename in folder:
		imaget = cv.imread("output/"+filename+".jpg", flags= cv.IMREAD_GRAYSCALE)
		imaget = cv.bitwise_not(imaget)
		imaget= imaget.reshape(28,28)
		imaget=imaget.reshape(1, 784)
		imaget= imaget.astype("float32")/255
		data.append(imaget)

	print(f"width: {width}, height:{height}")

	probabilities, labels = get_prob_array(bgty_model, data)

	labels.append(labels.index(0))

	print(f"width: {width}, height:{height}")
	print(f"Final array: {labels}")

	return labels, width, height

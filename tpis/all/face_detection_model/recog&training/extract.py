import os
import cv2
import argparse
import imutils
from imutils import paths
import numpy as np
import pickle

params = argparse.ArgumentParser()
params.add_argument("-ds", "--data", required=True)
params.add_argument("-e", "--embed", required=True)
params.add_argument("-d", "--det", required=True)
params.add_argument("-em", "--embmo", required=True)
params.add_argument("-co", "--conf", type=float, default=0.5)
args = vars(params.parse_args())

# import face detection files
protDir = os.path.sep.join([args["det"], "deploy.prototxt"])
modDir = os.path.sep.join([args["det"], "res10_300x300_ssd_iter_140000.caffemodel"])
det = cv2.dnn.readNetFromCaffe(protDir, modDir)
em = cv2.dnn.readNetFromTorch(args["embmo"])

# import the paths to the input images in data
imgSrc = list(paths.list_images(args["data"]))

# import current list of extracted faces referenced names
listEmb = []
listNames = []

# loop until there are no more images
for (i, imgSrc) in enumerate(imgSrc):
	# take folder name file is located in and use that as a nametag
	name = imgSrc.split(os.path.sep)[-2]

	# pull image file, resize its width and take its dimensions
	img = cv2.imread(imgSrc)
	img = imutils.resize(img, width=600)
	(h, w) = img.shape[:2]

	# create image blob
	imgBFI = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# use opencv's face detector to locate faces
	det.setInput(imgBFI)
	dets = det.forward()

	# if a face is found, continue, if not go onto the next image file
	if len(dets) > 0:
		# only apply bounding box to object with highest probability of being a face
		i = np.argmax(dets[0, 0, :, 2])
		conf = dets[0, 0, i, 2]

		# only locate objects that are considered highly confident to be faces
		if conf > args["conf"]:
			# find coords of the face
			box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# find the region of the image the face is in for the bounding box
			face = img[startY:endY, startX:endX]

			# create blob in the region of the image the face is located
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			em.setInput(faceBlob)vec = em.forward()

			# attribute folder name as the name associated with this face
			listNames.append(name)
			listEmb.append(vec.flatten())

# send extracted facial data to pickle file
data = {"embed": listEmb, "names": listNames}
f = open(args["embed"], "wb")
f.write(pickle.dumps(data))
f.close()
import os
from cv2 import *
import imutils
from imutils.video import FileVideoStream
import argparse
import pickle
import numpy as np

params = argparse.ArgumentParser()
params.add_argument("-d", "--det", required=True) # path to dir of faces
params.add_argument("-m", "--emb", required=True) # path to face embeddings/reference faces
params.add_argument("-r", "--rec", required=True) # face det
params.add_argument("-l", "--le", required=True) # face embedding model
params.add_argument("-c", "--conf", type=float, default=0.7) # minimum conf required to display bounding box
params = vars(params.parse_args())

protDir = os.path.sep.join([params["det"], "deploy.prototxt"]) # load face detecting files
# https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt

modDir = os.path.sep.join([params["det"],"res10_300x300_ssd_iter_140000.caffemodel"]) # load OpenCV face detecting files
# https://github.com/opencv/opencv/tree/master/modules/dnn

det = cv2.dnn.readNetFromCaffe(protDir, modDir) # load OpenCV face detecting files
# https://docs.opencv.org/trunk/d6/d0f/group__dnn.html

emd = cv2.dnn.readNetFromTorch(params["emb"]) # loading emd, for extracting faces
# https://docs.opencv.org/trunk/d6/d0f/group__dnn.html
# https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7

# https://docs.python.org/3/library/pickle.html
rec = pickle.loads(open(params["rec"], "rb").read()) # face rec PICKLE file in binary
le = pickle.loads(open(params["le"], "rb").read()) # le PICKLE file has names mapped to faces (rb = read/write in binary)

mediaIn = FileVideoStream("JimmyAndTom.mp4").start() # both

'''
framecounter = 0
if(framecounter % 5 == 0):
        resize the frame
        locate face(s)
        create boundingbox & get identified name
        find what the faces found resemble the most
        display frame + bounding boxes

else:
        just resize the frame
        display frame
framecounter += 1
'''

framecounter = 0 # frame iterator
while True:
        
        if(framecounter % 5 == 0): # only process every other frame to reduce computational expence
                frame = mediaIn.read() # read media
                frame = imutils.resize(frame, width=1080) # resize the frame
                (h, w) = frame.shape[:2] # aspect ratio of frame
         
                imgDFI = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(100.0, 170.0, 100.0), swapRB = False) # create image blob
                # blobs allow for easier face detection (facial features become more pronounced)
                # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
                # https://makehardware.com/2016/05/19/blob-detection-with-python-and-opencv/
                # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
                # https://developers.arcgis.com/python/sample-notebooks/counting-features-in-satellite-images-using-scikit-image/
                # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
                # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html

                det.setInput(imgDFI) # use OpenCV face det to locate faces
                detections = det.forward()
                for i in range(0, detections.shape[2]): # loop through in case there are more than 1 faces
                        conf = detections[0, 0, i, 2] # calculate conf of face match
                        if conf > params["conf"]: # decline anything below default argument value (0.7 or 70% confidence)
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # x,y coords for bounding box(es)
                                (startX, startY, endX, endY) = box.astype("int")

                                face = frame[startY:endY, startX:endX] # find ROI  (region of image)

                                # make blob from image (for ROI), send it to face emd
                                fBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True)
                                emd.setInput(fBlob) # use new blob as input for face emd
                                faceEmbed = emd.forward()

                                # send image through rec
                                predict = rec.predict_proba(faceEmbed)[0] # predict face matches
                                # https://www.scipy-lectures.org/packages/scikit-learn/index.html
                                
                                predMax = np.argmax(predict) # select highest match
                                # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argmax.html
                                
                                name = le.classes_[predMax] # pull name of highest match from le.PICKLE
                                # https://docs.python.org/2/library/pickle.html
        
                                # draw bounding boxes
                                cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 255, 255), 1) # bounding box
                                # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html?highlight=rectangle#cv2.rectangle

                                y = startY - 5 # sets text 5 px above bounding box
                                cv2.putText(frame, name, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # bounding box text
                                # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

                cv2.imshow("Select Facial Recognition", frame)        
                key = cv2.waitKey(27) & 0xFF
                if key == 27:
                        break
        else:
                frame = mediaIn.read()
                frame = imutils.resize(frame, width=1080)
                (h, w) = frame.shape[:2]
                cv2.imshow("Select Facial Recognition", frame)
                
        framecounter += 1

cv2.destroyAllWindows()
mediaIn.stop()

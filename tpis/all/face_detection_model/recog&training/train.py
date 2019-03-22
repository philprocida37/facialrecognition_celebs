import argparse
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

params = argparse.ArgumentParser()
params.add_argument("-em", "--emb", required=True)
params.add_argument("-re", "--recog", required=True)
params.add_argument("-l", "--le", required=True)
args = vars(params.parse_args())

# import embedding dump pickle file
data = pickle.loads(open(args["emb"], "rb").read())

# assign labels
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train recognizer
recogPickle = SVC(C=1.0, kernel="linear", probability=True)
recogPickle.fit(data["emb"], labels)

# create recognizer pickle
f = open(args["recog"], "wb")
f.write(pickle.dumps(recogPickle))
f.close()

# create label pickle
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
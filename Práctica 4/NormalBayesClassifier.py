import cv2

class NormalBayesClassifier():
    def __init__(self):
        self.model = cv2.NormalBayesClassifier()

    def train (self, samples, responses):
        self.model =cv2.NormalBayesClassifier()
        self.model.train(samples,responses)

    def predict (self, samples):
        retval,results = self.model.predict(samples)
        return results.ravel()
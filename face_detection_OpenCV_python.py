'''
Face_detection_by_OpenCV
Sean Rick Zhang
2019/3/11
'''
import cv2
import os
import argparse

class Face_detection(object):

    def __init__(self, save_path, model, XML_path, scaleFactor = 1.3, minNeighbors = 5, image_dir = None):
        """

        :param image_dir: customed image direction
        :param save_path: facial output direction
        :param model: "real_time" or "load_file"
        """

        self.image_dir = image_dir
        self.save_path = save_path
        self.model = model
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.path = XML_path
        if model == 'real_time':
            self.real_time()
        elif model == 'load_file':
            self.load_file()

    def load_file(self):
        img = cv2.imread(self.image_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_cascade = cv2.CascadeClassifier(self.path)
        faces = faces_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        result = []
        for (x, y, width, height) in faces:
            result.append((x, y, x + width, y + height))
        if result:
            num = 1
            for (x1, y1, x2, y2) in result:
                face_region = img[y1:y2, x1:x2]
                print("save picture " + str(num))

                cv2.imwrite(os.path.join(self.save_path, '{}-images.jpg'.format(num)), cv2.resize(face_region, (100, 100)))
                num+=1
        return num

    def real_time(self):
        cap = cv2.VideoCapture(0)
        num = 0
        while True:
            num+=1
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_cascade = cv2.CascadeClassifier(self.path)
            faces = faces_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
            result = []
            for (x, y, width, height) in faces:
                result.append((x, y, x + width, y + height))
            if result:
                for (x1, y1, x2, y2) in result:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), thickness=8)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                #press 'e' to EXIT
                break

        cap.release()
        cv2.destroyAllWindows()

def main(config):
    Face_detection(save_path=config.save_path, model=config.model, XML_path=config.XML_path,
                   scaleFactor=config.scaleFactor, minNeighbors=config.minNeighbors, image_dir=config.image_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--XML_path', type=str, default=None)
    parser.add_argument('--scaleFactor', type=float, default=1.3)
    parser.add_argument('--minNeighbors', type=int, default=5)
    parser.add_argument('--image_dir', type=str, default=None)

    config = parser.parse_args()
    main(config)
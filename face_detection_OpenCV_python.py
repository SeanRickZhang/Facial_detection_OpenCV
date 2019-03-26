'''
Face_detection_by_OpenCV
Sean Rick Zhang
2019/3/11
'''
import cv2
import os
import argparse

class Face_detection(object):

    def __init__(self, save_path, model, scaleFactor = 1.3, minNeighbors = 5, image_dir = None):
        """

        :param image_dir: customed image direction
        :param save_path: facial output direction
        :param model: "real_time" , "split_face" or "split_eye_mouth"
        """

        self.image_dir = image_dir
        self.save_path = save_path
        self.model = model
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        if model == 'real_time':
            self.real_time()
        elif model == 'split_face':
            self.split_face()
        elif model == 'split_eye_mouth':
            self.split_eye_mouth()

    def split_face(self):
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
            faces_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
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

    def split_eye_mouth(self):

        img = cv2.imread(self.image_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        face_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_files/haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_files/haarcascade_eye.xml'))
        mouth_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_files/haarcascade_mcs_mouth.xml'))

        faces = face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        result = []
        for (x, y, width, height) in faces:
            result.append((x, y, x + width, y + height))
        if result:
            face_num = 1
            for (x1, y1, x2, y2) in result:
                face_region = gray[y1:y2, x1:x2]
                alpha = 0.6
                y_apha = (int)((y2 - y1) * alpha + y1)
                upper_part = gray[y1:y_apha, x1:x2]
                lower_part = gray[y_apha:y2, x1:x2]
                eyes = eye_cascade.detectMultiScale(upper_part, self.scaleFactor, self.minNeighbors)
                eye_result = []
                for (eye_x, eye_y, eye_width, eye_height) in eyes:
                    eye_result.append((eye_x, eye_y, eye_x + eye_width, eye_y + eye_height))
                if eye_result:
                    eye_num = 1
                    for (eye_x1, eye_y1, eye_x2, eye_y2) in eye_result:
                        eye_region = img[y1 + eye_y1:y1 + eye_y2, x1 + eye_x1:x1 + eye_x2]
                        # eye_region = img[eye_y1:eye_y2, eye_x1:eye_x2]
                        cv2.imwrite(os.path.join(self.save_path, 'eye-image{}.jpg'.format(eye_num)),
                                    cv2.resize(eye_region, (64, 64)))
                        eye_num += 1
                mouthes = mouth_cascade.detectMultiScale(lower_part, self.scaleFactor, self.minNeighbors)
                mouth_result = []
                for (mouth_x, mouth_y, mouth_width, mouth_height) in mouthes:
                    mouth_result.append((mouth_x, mouth_y, mouth_x + mouth_width, mouth_y + mouth_height))
                if mouth_result:
                    mouth_num = 1
                    for (mouth_x1, mouth_y1, mouth_x2, mouth_y2) in mouth_result:
                        mouth_region = img[y_apha + mouth_y1:y_apha + mouth_y2, x1 + mouth_x1:x1 + mouth_x2]
                        # mouth_region = img[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

                        cv2.imwrite(os.path.join(self.save_path, 'mouth-image{}.jpg'.format(mouth_num)),
                                    cv2.resize(mouth_region, (64, 64)))
                        mouth_num += 1
                face_region = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(self.save_path, 'face-image{}.jpg'.format(face_num)), cv2.resize(face_region, (100, 100)))
                face_num += 1
                print("saved!")

def main(config):
    Face_detection(save_path=config.save_path, model=config.model, scaleFactor=config.scaleFactor, minNeighbors=config.minNeighbors, image_dir=config.image_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--scaleFactor', type=float, default=1.3)
    parser.add_argument('--minNeighbors', type=int, default=5)
    parser.add_argument('--image_dir', type=str, default=None)

    config = parser.parse_args()
    main(config)
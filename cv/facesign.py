
from __future__ import print_function
import cv2
import itertools

#https://github.com/Microsoft/Cognitive-Face-Python
endpoint = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'
key1 = 'cc0014b6b1934e2a92cd32c21de43fdb'
key2 = '93fc5ff7126641ab930a0a0568d75d09'


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture('video/facesign.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        self.out = cv2.VideoWriter('video/facesign_output.avi', fourcc, 30.0, (544, 960))

    def run(self):
        frame_idx = 0

        for _ in itertools.repeat(None, frame_idx):
            _, _ = self.cap.read()

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            vis = frame.copy()
            cv2.imshow('frame', vis)
            self.out.write(vis)

            frame_idx = frame_idx + 1

            ch = cv2.waitKey(1)
            if ch == 27:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    App().run()

import cognitive_face as CF
import cv2
import time
import json

#https://github.com/Microsoft/Cognitive-Face-Python
endpoint = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
key1 = 'cc0014b6b1934e2a92cd32c21de43fdb'
key2 = '93fc5ff7126641ab930a0a0568d75d09'
attributes = (
    'age,gender,headPose,smile,facialHair,glasses,emotion,hair,'
    'makeup,occlusion,accessories,blur,exposure,noise'
)

class App:
    def __init__(self):
        self.cap = cv2.VideoCapture('../video/facesign.mp4')

        CF.Key.set(key1)
        # If you need to, you can change your base API url with:
        CF.BaseUrl.set(endpoint)

    def run(self):
        frame_idx = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            path = '../image/facesign_%d.png' % frame_idx
            cv2.imwrite(path, frame)
            print(path)

            faces = CF.face.detect(path, False, False, attributes)
            frame_data.append({'frame_idx': frame_idx, 'faces': faces})

            print('frame_idx: %d' % frame_idx)
            print(faces)

            frame_idx = frame_idx + 1
            time.sleep(5) #secs

            break

        print("Frame count: %d" %  frame_idx)

        self.cap.release()

if __name__ == '__main__':
    frame_data = []
    App().run()
    with open('frame_data.json', 'w') as output:
        json.dump(frame_data, output, sort_keys=True, indent=4)

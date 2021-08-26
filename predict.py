import time
import cv2
import numpy as np
import torch
import argparse
from imutils.video import FPS, WebcamVideoStream

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def load_model(weights_path, device=""):
    half = device.type != "cpu"
    model = attempt_load(weights=weights_path, map_location=device)
    if half:
        model.half()
    stride = int(model.stride.max())
    return model, stride


class Predictor:
    def __init__(self, model_path, device=""):
        self.device = select_device(device)
        print("Using {} for detection".format(self.device))
        self.model, self.stride = load_model(model_path, device=self.device)
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

    def preprocess(self, images):
        def process(img):
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            half = self.device.type != "cpu"
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0
            return img

        images = [letterbox(image, self.img_size, stride=self.stride, auto=True)[0] for image in images]
        images = list(map(process, images))
        images = torch.stack(images)

        return images

    @torch.no_grad()
    def predict(self, imgs):
        all_boxes = []
        images = self.preprocess(imgs)

        pred = self.model(images, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=1000)

        # Each image
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(images.shape[2:], det[:, :4], imgs[0].shape).round()
                all_boxes.append(det.cpu().numpy())
            else:
                all_boxes.append([])
        return all_boxes

    def draw_box(self, image, boxes):
        for box in boxes:
            x0, y0, x1, y1 = [int(i) for i in box[:4]]
            conf = box[4]
            cls = int(box[5])
            cv2.rectangle(image, (x0, y0), (x1, y1), COLORS[cls % 3])
            text = "{}% {}".format(round(conf * 100, 1), predictor.names[cls])
            cv2.putText(image, text, (x0, y0), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def predict_webcam(self):
        print("[INFO] starting threaded video stream...")
        stream = WebcamVideoStream(src=0).start()  # default camera
        time.sleep(1.0)
        while True:
            frame = stream.read()
            key = cv2.waitKey(1) & 0xFF
            fps.update()
            boxes = self.predict([frame])[0]
            frame = self.draw_box(frame, boxes)
            # keybindings for display
            if key == ord('p'):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', frame)
                    if key2 == ord('p'):  # resume
                        break
            cv2.imshow('frame', frame)
            if key == 27:  # exit
                break
        return stream

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        saver = cv2.VideoWriter('data/demo/output.avi',   # video_path[:-4] + '-output' + video_path[-4:]
                                cv2.VideoWriter_fourcc(*'XVID'), 20,
                                (frame_width, frame_height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                boxes = self.predict([frame])[0]
                frame = self.draw_box(frame, boxes)
                saver.write(frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--image', default=False,
                    type=bool, help='Detect one image or list images')
parser.add_argument('--webcam', default=False,
                    type=bool, help='Detect by open webcam')
parser.add_argument('--video', default=False,
                    type=bool, help='Detect one video')
args = parser.parse_args()

if __name__ == "__main__":
    weight = 'weights/best.pt'
    predictor = Predictor(weight)

    if args.image:
        image_paths = ['data/demo/neymar.jpg']
        imgs = [cv2.imread(path) for path in image_paths]
        all_boxes = predictor.predict(imgs=imgs)

        for path, boxes in zip(image_paths, all_boxes):
            image = cv2.imread(path)
            image = predictor.draw_box(image, boxes=boxes)
            cv2.imwrite(path[:-4] + '-output' + path[-4:], image)

    if args.webcam:
        fps = FPS().start()
        stream = predictor.predict_webcam()
        fps.stop()

        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        stream.stop()

    if args.video:
        video_path = 'data/demo/messi.mp4'
        predictor.predict_video('data/demo/messi.mp4')

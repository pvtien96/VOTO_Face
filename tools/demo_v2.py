from threading import Thread
import cv2
import sys 
import numpy as np
from loguru import logger
from functools import wraps
import asyncio

sys.path.append('..')

# from multiprocess import Process, Queue
from queue import Queue
from threading import Thread

from models import get_model
from utils import normalize, square_crop, norm_crop
from module import DatabaseHandler, Searcher


class FaceDemo:
    def __init__(self, cfg, args, video_id=0):
        """Init the FaceDemo class
        Args:
            cfg: (fvcore.common.CfNode) Config for model
            args: (argparse.parser) Argument
            video_id: (int | str) id of webcam or path to video file
        """
        self.extractor = get_model('arcface')
        self.detector = get_model('scrfd')
        self.searcher = Searcher(distance='IP')
        self.cfg = cfg
        self.font = cv2.QT_FONT_NORMAL

        db = DatabaseHandler(self.searcher, cfg, self.detector, self.extractor)
        db.prepare()

        self.detector = get_model('mnet_cov2')
        self.detector.prepare()

        self.queue_buffer_size = args.queue_buffer_size
        self.max_frame_rate = args.max_frame_rate
        self.min_box_size = args.min_box_size
        self.recognize_wait = args.recognize_wait

        self.camera = cv2.VideoCapture(video_id)
        self.frame_queue = Queue(args.queue_buffer_size)
        self.suspicion_face_queue = Queue(args.max_face_number)
        self.person_queue = Queue(args.queue_buffer_size)

        self.gal_face_path = 'Unkown'
        self.__get_split_key()
        self.mask = False
        self.distance = 0.0
        self.person_queue.put((self.gal_face_path, None, self.distance))

        self.unkown_avatar = cv2.imread("./avatar_img/unkown.jpg")
        self.unkown_avatar = cv2.resize(self.unkown_avatar, (112, 112))

        self.mask_avatar = cv2.imread("./avatar_img/mask.png")
        self.mask_avatar = cv2.resize(self.mask_avatar, (112, 112))

    async def _camera_loop(self):
        """
        Read frame procees
        Please not call this function outside the class
        """
        loop = asyncio.get_running_loop()
        while True:
            start_time = loop.time()
            frame = self.camera.read()[1]

            frame = cv2.flip(frame, 1)
            self.frame_queue.put(frame)

            restime = (1 / self.max_frame_rate) - loop.time() + start_time
            if restime > 0:
                await asyncio.sleep(restime)

    async def _detection_loop(self):
        """
        Detection process
        Please not call this function outside the class
        """
        loop = asyncio.get_running_loop()
        i = 0

        while True:
            start = loop.time()
            frame_list = []
            i += 1

            frame = self.frame_queue.get()
            if i % self.recognize_wait == 0:
                frame = self.__detection_deal(frame, True)

            else:
                frame = self.__detection_deal(frame)

            logger.info(f'Detection cost: {loop.time() - start}')
            frame = self.__display(frame)

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

    async def _recognize_loop(self):
        """
        Recognize process
        Please not call this function outside the class
        """
        loop = asyncio.get_running_loop()
        while True:
            start_time = loop.time()
            self.__recognize_deal(self.suspicion_face_queue.get())
            logger.info(f"Recognize time {loop.time() - start_time}")

    def run(self):
        """
        Run demo
        """
        # Process(target=lambda: asyncio.run(self._detection_loop())).start()
        # Process(target=lambda: asyncio.run(self._recognize_loop())).start()
        # asyncio.run(self._camera_loop())
        t1 = Thread(target=lambda: asyncio.run(self._detection_loop())).start()
        t2 = Thread(target=lambda: asyncio.run(self._recognize_loop())).start()
        asyncio.run(self._camera_loop())

    def __recognize_deal(self, frame):
        embedding = self.extractor.get_embedding(frame)
        embedding = normalize(embedding)
        person_image, distance = self.searcher.search(embedding)

        if distance >= self.cfg.MODEL.Recognize.thresh:
            self.person_queue.put((person_image, frame, distance))
        else:
            self.person_queue.put(('Unkown', frame, distance))

    def __detection_deal(self, ori_frame, put_recorg=False):
        frame, scale = square_crop(ori_frame, self.cfg.MODEL.Detection.image_size[0])
        bboxes, landmarks = self.detector.detect(frame, self.cfg.MODEL.Detection.conf)

        # get largest box
        if len(bboxes) == 0:
            return ori_frame

        areas = []
        for i in range(bboxes.shape[0]):
            x = bboxes[i]
            area = (x[2] - x[0]) * (x[3] - x[1])
            areas.append(area)
        m = np.argsort(areas)[-1]
        bboxes = bboxes[m:m + 1]
        landmarks = landmarks[m:m + 1]

        bbox = bboxes[0]

        # Checking the size of bbounding box
        if bbox[2] - bbox[0] <= self.min_box_size  or bbox[3] - bbox[1] <= self.min_box_size:
            return ori_frame

        if put_recorg == True:
            rimg = norm_crop(frame, landmarks[0])
            self.suspicion_face_queue.put(rimg)
        if self.detector.masks is True:
            mask_score = bbox[5]
            # logger.info(f'Mask score: {mask_score}. Face score: {box[4]}')
            if mask_score <= self.cfg.MODEL.Detection.mask:
                color = (0, 255, 0)
                self.mask = False
            else:
                color = (0, 0, 255)
                self.mask = True
                cv2.putText(ori_frame,
                            "Please take off your mask",
                            (20, ori_frame.shape[0] - 20), self.font, 0.7,
                            (0, 0, 255))

        if self.mask:
            text = "Mask"
        elif self.gal_face_path == "Unkown":
            text = self.gal_face_path
        else:
            text = self.gal_face_path.split(self.split_key)[-2]

        pt1 = tuple(map(int, bbox[0:2] * scale))
        pt2 = tuple(map(int, bbox[2:4] * scale))
        cv2.rectangle(ori_frame, pt1, pt2, color, 1)
        cv2.putText(
            ori_frame,
            text,
            (pt1[0], pt1[1] - 60),
            self.font,
            0.7,
            (255, 255, 0)
        )
        cv2.putText(
            ori_frame,
            "Detect-Conf: {:0.2f} %".format(bbox[4] * 100),
            (pt1[0], pt1[1] - 40),
            self.font,
            0.7,
            color
        )

        cv2.putText(
            ori_frame,
            "Emb-Dist: {:0.2f}".format(self.distance),
            (pt1[0], pt1[1] - 20),
            self.font,
            0.7,
            color
        )

        return ori_frame

    def __display(self, frame):
        if not self.person_queue.empty():
            self.gal_face_path, self.current_face, self.distance = self.person_queue.get()

        if self.current_face is not None:
            frame[0:112, 0:112] = self.current_face
            frame = cv2.putText(frame, "Subject", (0, 10), self.font, 0.5, (85, 85, 255))

            if self.mask:
                frame[112:112+112, 0:112] = self.mask_avatar
            elif self.gal_face_path == "Unkown":
                frame[112:112+112, 0:112] = self.unkown_avatar
            else:
                gal_face = cv2.imread(self.gal_face_path)
                frame[112:112+112, 0:112] = gal_face
                frame = cv2.putText(frame, "GalleryMatch", (0, 112 + 10), self.font, 0.5, (85, 85, 255))

        return frame

    def __get_split_key(self):
        if sys.platform == 'win32':
            self.split_key = '\\'
        else:
            self.split_key = '/'


if __name__ == "__main__":
    from configs import update_config
    import argparse

    parser = argparse.ArgumentParser(description='Demo')

    # =================== General ARGS ====================
    parser.add_argument('--max_face_number',
                        type=int,
                        help='Max face number',
                        default=16)
    parser.add_argument('--max_frame_rate',
                        type=int,
                        help='Max frame rate',
                        default=25)
    parser.add_argument('--queue_buffer_size',
                        type=int,
                        help='MP Queue size',
                        default=12)
    parser.add_argument('-c', '--usb_camera_code',
                        type=int,
                        nargs='+',
                        help='Code of usb camera. (You can use media file path to test with videos.)',
                        default=0)
    parser.add_argument('--min_box_size',
                        type=int,
                        default=100,
                        help='Min size of the bbouding box')
    parser.add_argument('--recognize_wait',
                        type=int,
                        default=100,
                        help='Sleep recognize process for num of iteration' +
                        ' (Recomment not to use small number)')

    args = parser.parse_args()
    cfg = update_config('../configs/config.yaml', [])

    demo = FaceDemo(cfg, args, args.usb_camera_code)
    demo.run()
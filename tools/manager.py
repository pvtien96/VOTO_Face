from threading import Thread
from queue import Queue
import socket
import argparse
import sys
import cv2
from loguru import logger
import base64
import numpy as np
import asyncio
from functools import wraps

sys.path.append('..')

from configs import update_config
from models import get_model
from utils import normalize, square_crop, norm_crop
from module import DatabaseHandler, Searcher

class FaceRecognition:
    def __init__(self, cfg, args):
        """
        Init the FaceRecognition class
        Args:
            cfg: (fvcore.common.CfNode) Config for model
            args: (argparse.parser) Argument
        """
        self.extractor = get_model('arcface')
        self.detector = get_model('mnet_cov2')
        self.searcher = Searcher(distance='IP')
        self.cfg = cfg
        self.font = cv2.QT_FONT_NORMAL

        self.receiveSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.receiveSocket.bind(('', self.cfg.IO.receivePort))

        dbHander =DatabaseHandler(self.searcher, cfg, self.detector, self.extractor)
        dbHander.prepare()
        self.detector.prepare()

        self.queue_buffer_size = args.queue_buffer_size
        self.max_frame_rate = args.max_frame_rate
        self.min_box_size = args.min_box_size

        self.frame_queue = Queue(args.queue_buffer_size)
        self.suspicion_face_queue = Queue(args.max_face_number)
        self.person_queue = Queue(args.queue_buffer_size)
    

        self.gal_face_path = 'Unkown'
        self.__get_split_key()
        self.distance = 0.0
        self.person_queue.put((self.gal_face_path, None, self.distance))

        self.unkown_avatar = cv2.imread("./avatar_img/unkown.jpg")
        self.unkown_avatar = cv2.resize(self.unkown_avatar, (112, 112))

    async def _receive_loop(self):
        """
        Listen UDP stream
        """
        loop = asyncio.get_running_loop()
        while (True):
            message, address = self.receiveSocket.recvfrom(self.cfg.IO.receiveBufferSize)
            frame_received = base64.decodebytes(message)
            frame_as_np = np.frombuffer(frame_received, dtype=np.uint8)
            frame = cv2.imdecode(frame_as_np, flags=1)

            self.frame_queue.put(frame)
        
    async def _detection_loop(self):
        """
        Detection process
        Do not call this function outside the class
        """
        loop = asyncio.get_running_loop()
        i = 0

        while True:
            start = loop.time()
            frame_list = []
            i += 1

            frame = self.frame_queue.get()
            frame = self.__detection_deal(ori_frame=frame, put_recorg=True)

            logger.info(f'Detection cost: {loop.time() - start}')
            frame = self.__display(frame)

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

    async def _recognize_loop(self):
        """
        Recognize process
        Do not call this function outside the class
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
        t1 = Thread(target=lambda: asyncio.run(self._detection_loop())).start()
        t2 = Thread(target=lambda: asyncio.run(self._recognize_loop())).start()
        asyncio.run(self._receive_loop())

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
        
        if self.gal_face_path == "Unkown":
            text = self.gal_face_path
        else:
            text = self.gal_face_path.split(self.split_key)[-2]

        color = (0, 255, 0)
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

            if self.gal_face_path == "Unkown":
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
    parser.add_argument('--min_box_size',
                        type=int,
                        default=100,
                        help='Min size of the bbouding box')
    args = parser.parse_args()
    cfg = update_config('../configs/config.yaml', [])

    demo = FaceRecognition(cfg, args)
    demo.run()

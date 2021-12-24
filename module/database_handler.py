"""
This file implement the database handler for updating database when the program
is initialized and this class will be deleted when the database checker is
finished

* Note:
    + The format of image database must be:
        ./database_image/
            ./persons1/
                ./persons1_{some_thing}.png
                ./persons1_{some_thing}.png
            ./persons2/
                ./persons2_{some_thing}.png
                ./persons2_{some_thing}.png

    + The format of feature database must in the format:
        ./database_feature/
            ./persons1/
                ./fea_persons1_{some_thing}.npy
                ./fea_persons1_{some_thing}.npy
            ./persons2/
                ./fea_persons2_{some_thing}.npy
                ./fea_persons2_{some_thing}.npy
"""
import glob
import os
import sys
import numpy as np

import cv2
from loguru import logger

from utils import normalize, square_crop, norm_crop


class DatabaseHandler:
    def __init__(self, searcher, cfg, detector, extractor):
        """Init DatabaseHandler class

        - Check the difference between image_path and feature_path, then update
        the feature database.
        - Add feature of database to searcher which used for recognize task.

        Args:
            searcher: (class: Searcher) faiss searcher class which implement in
                    file "searcher.py"
            cfg: (dist) config dictionary
            detector: (model) detection method (recomment to use scrfd)
            extractor: (model) feature extractor method (recomment to use scrfd)
        """
        self.cfg = cfg
        self.image_path = self.cfg.DATABASE.image_path
        self.feature_path = self.cfg.DATABASE.feature_path
        self.searcher = searcher
        self.remove_index = []

        self.detector = detector
        self.extractor = extractor

        # Checking database
        self._check_database()
        # Update database if need
        self._update()
        # Remove unuse image
        self._remove()

    def prepare(self):
        """
        Prepare database by add person and it feature into searcher.
        """
        for i in range(len(self.feature_database)):
            feature = np.load(self.feature_database[i], allow_pickle=True)
            self.searcher.add_one(feature, self.image_database[i])

    def _check_database(self):
        """
        Check not extract image in database
        """
        logger.info("Checking current database")
        self.image_database = glob.glob('{}/*/**'.format(self.image_path))
        self.feature_database = [
            self.__get_fea_path(x) for x in self.image_database
        ]

        feature_temp = glob.glob('{}/*/**'.format(self.feature_path))
        self.update_index = [
            index for index in range(len(self.feature_database))
            if self.feature_database[index] not in feature_temp
        ]

    def _update(self):
        """
        Update feature database
        """
        if len(self.update_index) == 0:
            return None

        logger.info("Found {} new image.".format(len(self.update_index)))

        for index in self.update_index:
            image = cv2.imread(self.image_database[index])
            # image = ImageData(image, self.cfg.MODEL.Detection.image_size)
            if image.shape[:2] != tuple(self.cfg.MODEL.Recognize.image_size):
                image, _ = square_crop(image,
                                       self.cfg.MODEL.Detection.image_size[0])
                self.detector.prepare()
                try:
                    # If using scrfd method, using the built in get largest
                    # box which implement in it class.
                    bboxes, landmarks = self.detector.detect(
                        image,
                        max_num=self.cfg.SCRFD.max_num,
                        metric=self.cfg.SCRFD.metric)
                except:
                    # Else using RetinaFace
                    bboxes, landmarks = self.detector.detect(image)

                    if len(bboxes):
                        areas = []
                        for i in range(bboxes.shape[0]):
                            x = bboxes[i]
                            area = (x[2] - x[0]) * (x[3] - x[1])
                            areas.append(area)
                        m = np.argsort(areas)[-1]
                        bboxes = bboxes[m:m + 1]
                        landmarks = landmarks[m:m + 1]

                # Add non face or low quality image to remove list
                if len(bboxes) == 0:
                    self.remove_index.append(index)
                    continue

                # Crop the image
                bbox = bboxes[0]
                rimg = norm_crop(image, landmarks[0])
                cv2.imwrite(self.image_database[index], rimg)

            else:
                rimg = image

            fea = self.extractor.get_embedding(rimg)
            fea = normalize(fea)

            # Check exist person folder, if not ceate it
            if not os.path.exists(os.path.dirname(
                    self.feature_database[index])):
                os.mkdir(os.path.dirname(self.feature_database[index]))

            np.save(self.feature_database[index], fea)

        logger.success("Update database successfully !!!")

    def _remove(self):
        """
        Remove unuse image database
        """
        if len(self.remove_index) == 0:
            return None

        logger.info("Remove {} non-face or low quality image".format(len(self.remove_index)))

        # Remove form database list
        remove_image = [self.image_database[i] for i in self.remove_index]
        remove_feature = [self.feature_database[i] for i in self.remove_index]
        self.image_database = [item for item in self.image_database if item not in remove_image]
        self.feature_database = [item for item in self.feature_database if item not in remove_feature]

        # Remove image from database folder
        for remove_item in remove_image:
            os.remove(remove_item)


    def __get_fea_path(self, text):
        if sys.platform == 'win32':
            split_key = "\\"
        else:
            split_key = '/'

        persons = text.split(split_key)[-2]
        image_name = text.split(split_key)[-1].split('.')[0]
        return os.path.join(self.feature_path, persons,
                            'fea_{}.npy'.format(image_name))
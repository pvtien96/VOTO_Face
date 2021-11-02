import faiss
from loguru import logger


class Searcher():
    def __init__(self, fea_dim=512, distance='L2'):
        """
        Searcher for Face Recorgnize
        params: fea_dim: dimensions of embeeding
        params: distance: distance type [L2, IP]
        """
        self.persons = []

        if distance == 'L2':
            self.index = faiss.IndexFlatL2(fea_dim)
        elif distance == 'IP':
            self.index = faiss.IndexFlatIP(fea_dim)
        else:
            logger.exception("The distance {distance} is not support yet !!!")

    def add_one(self, fea, person):
        self.index.add(fea)
        self.persons.append(person)

    def add_all(self, fea_list, persons_list):
        for i in range(len(fea_list)):
            self.add_one(fea_list[i], persons_list[i])

    def search(self, query):
        D, I = self.index.search(query, 1)

        return self.persons[I[0][0]], D[0][0]

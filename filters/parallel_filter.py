from filters.filter import Filter
from logging import Logger
import multiprocessing as mp

class ParallelFilter(Filter):
    def __init__(self, name: str, logger: Logger):
        super().__init__(name, logger)
        self.filters = []

    def add_filter(self, filter: Filter):
        """
        Adds a filter to the parallel filter.
        :param filter: A filter object that takes an image as input and returns the processed image.
        """
        self.filters.append(filter)

    def apply(self, image):
        """
        Applies all filters in parallel to the input image.
        :param image: The image to process.
        :return: The final processed image.
        """
        # Implement parallel execution of filters here
        mp_pool = mp.Pool(processes=len(self.filters))
        results = mp_pool.map(lambda f: f.apply(image), self.filters)
        mp_pool.close()
        mp_pool.join()

        return results
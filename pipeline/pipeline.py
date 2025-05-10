class Pipeline:
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_function : callable):
        """
        Adds a filter to the pipeline.
        :param filter_function: A callable function or filter class that takes an image as input and returns the processed image.
        """
        self.filters.append(filter_function)

    def add_parallel_filter(self, filter_function):
        """
        Adds a filter to the pipeline that will be executed in parallel.
        Pokud to má vůbec smysl?
        """
        return NotImplementedError("Parallel filter execution is not implemented yet.")
    
    def execute(self, image):
        """
        Executes the pipeline of filters on the input image.
        :param image: The image to process.
        :return: The final processed image.
        """
        for filter_function in self.filters:
            image = filter_function(image)
        return image

from filters.filter import Filter

class Pipeline:
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter : Filter):
        """
        Adds a filter to the pipeline.
        :param filter_function: A callable function or filter class that takes an image as input and returns the processed image.
        """
        self.filters.append(filter)
    
    def execute(self, image):
        """
        Executes the pipeline of filters on the input image.
        :param image: The image to process.
        :return: The final processed image.
        """
        for filter_function in self.filters:
            image = filter_function(image)

        #TODO - Final filter that returns data

        #TODO - Save data to DB

        #TODO - Return data
        return image
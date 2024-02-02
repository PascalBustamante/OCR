import cv2 as cv 
import numpy as np

class DataManipulator:
    def __init__(self, filename: str) -> None:
        self.data = DataSample
        self.
    
    def clean_data(self):
        pass

    def contrast(self):
        pass

    def skeletonization(self):
        pass
    

class DataSample:
    ""
    def __init__(self, filenames: [str], connection=None) -> None:
        self._filenames = filenames ## IDs for src based on idx?
        self._connection = connection
        self.data_block = []

        if connection:
            self.load()

            cv2.samples.findFile("lena.jpg")

    def __len__(self):
        return len(self._filepaths)
    
    def __getitem__(slef):
        pass
    
    def load(self):
        ''' Load an image from the specified file. '''
        for path in self._filepaths:
            try:
                src = cv.imread(self._filename)
                if src is not None:
                    # Perform additional processing if needed
                    self.data_block.append(src)
                else:
                    print(f"Error: Failed to load image from '{self._filename}'.")

            except Exception as e:
                print(f"Error: Failed to load image from '{self._filename}': {e}")

    def save(self, dirname: str, image):
        ''' Store an image to the specified file. '''
        for idx in range(len(self.data_block)):
            filename = 
            path = dirname + 
            try:
                cv.imwrite(path, image)
                print(f"Image successfully saved to '{path}'.")
            except Exception as e:
                print(f"Error: Failed to save image to '{path}': {e}")

    def get_filename(self):
        ''' Get the filename associated with this data sample. '''
        return self._filename

    def set_filename(self, new_filename):
        ''' Set a new filename for this data sample. '''
        self._filename = new_filename

    def get_connection(self):
        ''' Get the database connection associated with this data sample. '''
        return self._connection

    def set_connection(self, new_connection):
        ''' Set a new database connection for this data sample. '''
        self._connection = new_connection
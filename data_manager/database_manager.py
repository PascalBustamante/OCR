import psycopg2
from psycopg2 import sql
import cv2 as cv

class DatabaseManager:
    def __init__(self, db_credentials=None) -> None:
        self._db_credentials = db_credentials
        self._connection = None

        if db_credentials:
            self.connect()

    def connect(self):
        try:
            # Establish a connection to the PostgreSQL database
            self._connection = psycopg2.connect(**self._db_credentials)
            print("Connected to the database.")
        except Exception as e:
            print("Error: Unable to connect to the database.", e)

    def disconnect(self):
        try:
            if self._connection is not None:
                self._connection.close()
                print("Disconnected from the database.")
        except Exception as e:
            print("Error: Unable to disconnect from the database.", e)

    def execute_query(self, query, parameters=None):
        try:
            cursor = self._connection.cursor()

            if parameters is not None:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            # Fetch the result if needed
            result = cursor.fetchall()

            cursor.close()
            return result

        except Exception as e:
            print(f"Error: Unable to execute query. {e}")
            return None
    
    def get_connection(self):
        ''' Get the database connection. '''
        return self._connection
    
    def set_credential(self, db_credentials):
        self._db_credentials = db_credentials

class DataSample:
    ""
    def __init__(self, filenames: [str], connection=None) -> None:
        self._filenames = []
        self._connection = connection
        self.data_block = []

        if connection:
            self.load()

    def __len__(self):
        return len(self._filenames)
    
    def __getitem__(self, identifier):
        if type(identifier) == int: 
            return self.data_block[identifier], self._filenames[identifier] 
        
        elif type(identifier) == str:
            idx = self._filenames.index(identifier)
            return self.data_block[idx], self._filenames[idx]

    
    def load(self, filenames):
        ''' Load an image from the specified file. '''
        for filename in filenames:
            self._add_data(filename)

    def save(self, dirname: str, image):
        ''' Store an image to the specified file. '''
        for idx in range(len(self.data_block)):
            filename = self._filenames[idx]
            data = self.data_block[idx]    ## Double check data type, and add some tests/constrainsts
            path = dirname + filename
            try:
                cv.imwrite(path, data)
                print(f"Image successfully saved to '{path}'.")
            except Exception as e:
                print(f"Error: Failed to save image to '{path}': {e}")

    def get_filenames(self):
        ''' Get the filename associated with this data sample. '''
        return self._filenames

    def _add_data(self, new_filename):
        ''' Set a new filename for this data sample. '''
        try:
            src = cv.imread(cv.samples.findFile(new_filename))
            if src:
                # Perform additional processing if needed
                self.data_block.append(src)
                self._filenames.append(new_filename) 

            else:
                print(f"Error: Failed to load image from '{new_filename}'.")

        except Exception as e:
            print(f"Error: Failed to load image from '{new_filename}': {e}")


    def get_connection(self):
        ''' Get the database connection associated with this data sample. '''
        return self._connection

    def set_connection(self, new_connection):
        ''' Set a new database connection for this data sample. '''
        self._connection = new_connection
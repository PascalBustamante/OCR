import psycopg2
from psycopg2 import sql

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
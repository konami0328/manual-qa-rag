import os
from pprint import pprint
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError


class MongoConfig:
    # --- Default Configuration ---
    # Fetch database settings from environment variables with safe defaults
    _host = os.getenv("MONGO_HOST", "localhost")
    _port = int(os.getenv("MONGO_PORT", 27017))
    _db_name = os.getenv("MONGO_DB_NAME", "mydatabase")
    _username = os.getenv("MONGO_USERNAME")
    _password = os.getenv("MONGO_PASSWORD")
    _auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    # --- Connection Parameters ---
    _max_pool_size = 100
    _connect_timeout = 5000  # milliseconds
    _socket_timeout = 3000   # milliseconds

    # --- Singleton State ---
    _client = None
    _db = None

    @classmethod
    def _build_connection_uri(cls):
        """Constructs the MongoDB connection URI string."""
        if cls._username and cls._password:
            return f"mongodb://{cls._username}:{cls._password}@{cls._host}:{cls._port}/?authSource={cls._auth_source}"
        return f"mongodb://{cls._host}:{cls._port}"

    @classmethod
    def initialize(cls):
        """Initializes the MongoDB client and verifies the connection."""
        if cls._client is None:
            try:
                cls._client = MongoClient(
                    cls._build_connection_uri(),
                    maxPoolSize=cls._max_pool_size,
                    connectTimeoutMS=cls._connect_timeout,
                    socketTimeoutMS=cls._socket_timeout,
                    serverSelectionTimeoutMS=5000
                )

                # Verify connection health by pinging the server
                cls._client.admin.command('ping')
                cls._db = cls._client[cls._db_name]
                print("Successfully connected to MongoDB")

            except ConfigurationError as e:
                raise RuntimeError(f"MongoDB configuration error: {str(e)}")
            except ConnectionFailure as e:
                raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected MongoDB connection error: {str(e)}")

    @classmethod
    def get_db(cls):
        """Returns the database instance, initializing if necessary."""
        if cls._client is None:
            cls.initialize()
        return cls._db

    @classmethod
    def get_collection(cls, collection_name):
        """Returns a specific collection instance from the active database."""
        return cls.get_db()[collection_name]

    @classmethod
    def close(cls):
        """Closes the MongoDB client and resets the singleton state."""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("MongoDB connection closed")


# Initialize the global connection during application startup
MongoConfig.initialize()


if __name__ == "__main__":
    # Example usage and basic CRUD operations
    client = MongoConfig()
    collection = MongoConfig.get_collection("my_collection")
    
    # Insert a single document
    dic = {'name': 'serena', "id": 1532}
    collection.insert_one(dic)
    
    # Insert multiple documents
    list_of_records = [{'name': 'amy', 'id': 1798}, {'name': 'bob', 'id': 1631}]
    collection.insert_many(list_of_records)
    
    # Retrieve and display all documents in the collection
    for record in collection.find():
        pprint(record)
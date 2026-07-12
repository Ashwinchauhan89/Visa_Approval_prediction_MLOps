import os

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

from us_visa.constants import MONGODB_URL_KEY, DATABASE_NAME


class MongoDBClient:
    def __init__(self, database_name: str = DATABASE_NAME):
        self.database_name = database_name
        self.mongo_url = os.getenv(MONGODB_URL_KEY)
        if not self.mongo_url or MongoClient is None:
            self.client = None
            self.database = None
            return
        self.client = MongoClient(self.mongo_url)
        self.database = self.client[self.database_name]
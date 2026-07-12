import os
import sys
from pandas import DataFrame

from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException
from us_visa.utils.main_utils import load_object


class SimpleStorageService:
    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        os.makedirs(os.path.dirname(to_filename), exist_ok=True)
        with open(from_filename, "rb") as src, open(to_filename, "wb") as dst:
            dst.write(src.read())
        if remove:
            os.remove(from_filename)


class USvisaEstimator:
    def __init__(self, bucket_name, model_path):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model = None

    def is_model_present(self, model_path):
        return os.path.exists(model_path)

    def load_model(self) -> USvisaModel:
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(self.model_path)
            return load_object(self.model_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def save_model(self, from_file, remove: bool = False) -> None:
        try:
            self.s3.upload_file(from_file, to_filename=self.model_path, bucket_name=self.bucket_name, remove=remove)
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise USvisaException(e, sys)
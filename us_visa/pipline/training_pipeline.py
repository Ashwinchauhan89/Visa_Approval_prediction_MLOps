import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from us_visa.constants import TARGET_COLUMN
from us_visa.entity.artifact_entity import ClassificationMetricArtifact, DataIngestionArtifact, DataTransformationArtifact, ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact
from us_visa.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelEvaluationConfig, ModelPusherConfig, ModelTrainerConfig
from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, read_yaml_file, save_numpy_array_data, save_object, write_yaml_file


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def _resolve_source_data(self) -> pd.DataFrame:
        data_file = os.path.join("data", "usvisa.csv")
        if os.path.exists(data_file):
            return pd.read_csv(data_file)
        raise FileNotFoundError("Provide data/usvisa.csv before running training")

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self._resolve_source_data()
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None,
            )
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_df.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            return DataIngestionArtifact(self.data_ingestion_config.training_file_path, self.data_ingestion_config.testing_file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            schema = read_yaml_file("config/schema.yaml")
            train_df = pd.read_csv(data_ingestion_artifact.trained_file_path)
            expected = schema["dataset_columns"]["categorical_columns"] + schema["dataset_columns"]["numerical_columns"] + [schema["dataset_columns"]["target_column"]]
            actual = list(train_df.columns)
            validation_status = actual == expected
            report = {"validation_status": validation_status, "expected_columns": expected, "actual_columns": actual}
            write_yaml_file(os.path.join(self.data_transformation_config.data_transformation_dir, "validation_report.yaml"), report, replace=True)
            return report
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact):
        try:
            schema = read_yaml_file("config/schema.yaml")["dataset_columns"]
            numerical_columns = schema["numerical_columns"]
            categorical_columns = schema["categorical_columns"]

            train_df = pd.read_csv(data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(data_ingestion_artifact.test_file_path)

            target_map = {"Certified": 1, "Denied": 0}
            train_y = train_df[TARGET_COLUMN].map(target_map).fillna(train_df[TARGET_COLUMN]).astype(int)
            test_y = test_df[TARGET_COLUMN].map(target_map).fillna(test_df[TARGET_COLUMN]).astype(int)
            train_x = train_df.drop(columns=[TARGET_COLUMN])
            test_x = test_df.drop(columns=[TARGET_COLUMN])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_columns),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
                ]
            )

            train_arr = preprocessor.fit_transform(train_x)
            test_arr = preprocessor.transform(test_x)
            train_arr = np.c_[train_arr.toarray() if hasattr(train_arr, "toarray") else train_arr, train_y.to_numpy()]
            test_arr = np.c_[test_arr.toarray() if hasattr(test_arr, "toarray") else test_arr, test_y.to_numpy()]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            return DataTransformationArtifact(self.data_transformation_config.transformed_object_file_path, self.data_transformation_config.transformed_train_file_path, self.data_transformation_config.transformed_test_file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            train_arr = load_numpy_array_data(data_transformation_artifact.transformed_train_file_path)
            preprocessor = load_object(data_transformation_artifact.transformed_object_file_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            predictions = model.predict(x_train)
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(y_train, predictions),
                precision_score=precision_score(y_train, predictions),
                recall_score=recall_score(y_train, predictions),
            )
            trained_model = USvisaModel(preprocessor, model)
            save_object(self.model_trainer_config.trained_model_file_path, trained_model)
            return ModelTrainerArtifact(self.model_trainer_config.trained_model_file_path, metric_artifact)
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact):
        try:
            test_arr = load_numpy_array_data(self.data_transformation_config.transformed_test_file_path)
            trained_model = load_object(model_trainer_artifact.trained_model_file_path)
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
            trained_score = f1_score(y_test, trained_model.trained_model_object.predict(x_test))
            best_model_score = 0.0
            is_model_accepted = trained_score >= self.model_evaluation_config.changed_threshold_score
            return ModelEvaluationArtifact(is_model_accepted, trained_score - best_model_score, self.model_evaluation_config.s3_model_key_path, model_trainer_artifact.trained_model_file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            os.makedirs(os.path.dirname(self.model_pusher_config.s3_model_key_path), exist_ok=True)
            save_object(self.model_pusher_config.s3_model_key_path, load_object(model_evaluation_artifact.trained_model_path))
            return ModelPusherArtifact(self.model_pusher_config.bucket_name, self.model_pusher_config.s3_model_key_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact, model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted.")
                return None
            self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        except Exception as e:
            raise USvisaException(e, sys)
import os
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan!")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File {path} kosong!")
    if 'Churn' not in df.columns:
        raise ValueError(f"Kolom 'Churn' tidak ditemukan di {path}!")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype('float64')
    X = X.fillna(0)
    return X, y

def train_and_log(train_path, test_path, model_path, report_path):
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    pd.DataFrame(report).to_csv(report_path)
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Evaluation report saved to {report_path}")

    # Gunakan localhost untuk MLflow tracking
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    experiment_name = "Modelling_Cecilia-Agnes-Vechrisda-Manalu"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None or experiment.lifecycle_stage == "deleted":
        experiment_name = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        client.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    input_example = X_test.iloc[[0]]
    signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run(run_name="RandomForest-Train"):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_artifact(report_path)
        logging.info("Model & report logged to MLflow.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_and_log(
        train_path = os.path.join(BASE_DIR, "namadataset_preprocessing", "train.csv"),
        test_path = os.path.join(BASE_DIR, "namadataset_preprocessing", "test.csv"),
        model_path = os.path.join(BASE_DIR, "model", "churn_rf.pkl"),
        report_path = os.path.join(BASE_DIR, "model", "eval_report.csv")
    )

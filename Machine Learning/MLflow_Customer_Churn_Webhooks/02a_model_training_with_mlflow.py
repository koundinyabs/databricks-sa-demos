# Databricks notebook source
# MAGIC %md
# MAGIC # Streamlining ML Operations Using Databricks
# MAGIC 
# MAGIC ##Step 2a - Model Training and Tracking

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read and Create Training + Test Datasets

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read data
data = spark.table("telco_churn.churn_features").toPandas().drop(["customerID"], axis=1)

train, test = train_test_split(data, test_size=0.30, random_state=206)
colLabel = 'churn'

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit model and log with MLflow
# MAGIC 
# MAGIC **Databricks Autologging is a no-code solution that extends MLflow automatic logging to deliver automatic experiment tracking for machine learning training sessions on Databricks.** 
# MAGIC 
# MAGIC With Databricks Autologging, model parameters, metrics, files, and lineage information are automatically captured when you train models from a variety of popular machine learning libraries. Training sessions are recorded as MLflow tracking runs. Model files are also tracked so you can easily log them to the MLflow Model Registry and deploy them for real-time scoring with MLflow Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC #### MLflow Auto Logging Config (Optional)

# COMMAND ----------

# Override default options
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=True
)

# Set experiment
mlflow.set_experiment("/Users/usman.zubair@databricks.com/telco_churn_experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fit Model and Track with MLflow

# COMMAND ----------

# Fit model
model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=32)
model.fit(train_x, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is an MLflow Model?
# MAGIC 
# MAGIC An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools—for example, batch inference on Apache Spark and real-time serving through a REST API. You can save all different types of models including: 
# MAGIC <br>
# MAGIC <br>
# MAGIC 
# MAGIC   - Python Function (python_function)
# MAGIC   - R Function (crate)
# MAGIC   - H2O (h2o)
# MAGIC   - Keras (keras)
# MAGIC   - MLeap (mleap)
# MAGIC   - PyTorch (pytorch)
# MAGIC   - Scikit-learn (sklearn)
# MAGIC   - Spark MLlib (spark)
# MAGIC   - TensorFlow (tensorflow)
# MAGIC   - ONNX (onnx)
# MAGIC   - MXNet Gluon (gluon)
# MAGIC   - XGBoost (xgboost)
# MAGIC   - LightGBM (lightgbm)
# MAGIC 
# MAGIC #### For this Python model, we used the ```mlflow.sklearn``` model type.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparamter Tuning with Spark
# MAGIC 
# MAGIC Databricks ML Runtimes offer an augmented version of [Hyperopt](https://github.com/hyperopt/hyperopt), a library for ML hyperparameter tuning in Python. It includes:
# MAGIC * Installed Hyperopt
# MAGIC * `SparkTrials` class, which provides distributed tuning via Apache Spark
# MAGIC * Automated MLflow tracking, configured via `spark.databricks.mlflow.trackHyperopt.enabled` (on by default)
# MAGIC 
# MAGIC The workflow with `Hyperopt` is as following:<br>
# MAGIC * Prepare the dateset.<br>
# MAGIC * Define a function to minimize.<br>
# MAGIC * Define a search space over hyperparameters.<br>
# MAGIC * Select a search algorithm.<br>
# MAGIC * Run the tuning algorithm with hyperopt `fmin()`.

# COMMAND ----------

import hyperopt as hp
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt import SparkTrials
from sklearn.model_selection import cross_val_score

# Databricks notebook source
# MAGIC %md
# MAGIC # Streamlining ML Operations Using Databricks
# MAGIC 
# MAGIC ##Step 1 - Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true" width="1400" height="2800">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Read Customer Churn Bronze Dataset from Data Lake into Dataframe

# COMMAND ----------

# Read into Spark
telcoDF = spark.table("telco_churn.bronze_customers")

display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define Featurication Function
# MAGIC 
# MAGIC Using `koalas` allows us to scale `pandas` code.  

# COMMAND ----------

# DBTITLE 0,Define featurization function
import databricks.koalas as ks

def compute_churn_features(data):
  
  # Convert to koalas
  data = data.to_koalas()
  
  # OHE
  data = ks.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],dtype = 'int64')
  
  # Convert label to int and rename column
  data['churnString'] = data['churnString'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churnString': 'int32'})
  data = data.rename(columns = {'churnString': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC ####Write Features to a Table

# COMMAND ----------

churn_features_df = compute_churn_features(telcoDF).to_spark()

churn_features_df.write.format("delta").mode("overwrite").save("/home/usman.zubair@databricks.com/ibm-telco-churn/churn_features/")

#Create Features Table
_ = spark.sql("""
    CREATE TABLE telco_churn.churn_features
    USING DELTA
    LOCATION '/home/usman.zubair@databricks.com/ibm-telco-churn/churn_features/'
    """)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM telco_churn.churn_features

# COMMAND ----------

# MAGIC %md
# MAGIC ####Write Features to the Databricks Feature Store
# MAGIC 
# MAGIC The feature store allows other data scientists and ML engineers to discover what has already been computed for machine learning models reducing duplicative work and creating consistency between models.  

# COMMAND ----------

# DBTITLE 0,Write features to the feature store
from databricks.feature_store import feature_table
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

churn_features_df = compute_churn_features(telcoDF)

churn_feature_table = fs.create_feature_table(
  name='telco_churn.churn_features_s',
  keys='customerID',
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the telco_churn.bronze_customers table in the lakehouse.  I created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=churn_features_df.to_spark(), name='telco_churn.churn_features_s', mode='overwrite')

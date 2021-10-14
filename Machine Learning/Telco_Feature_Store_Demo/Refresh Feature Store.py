# Databricks notebook source
# MAGIC %md
# MAGIC # Streamlining ML Operations Using Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ##Refresh Feature Store

# COMMAND ----------

# Read Data from Source
telcoDF = spark.table("telco_churn.bronze_customers")

display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Define Features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Demographic Features

# COMMAND ----------

from databricks.feature_store import feature_table

demographic_cols = ["customerID", "gender", "seniorCitizen", "partner", "dependents"]

@feature_table
def compute_demographic_features(data):
  return data.select(demographic_cols)

demographics_df = compute_demographic_features(telcoDF)

display(demographics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Service Features

# COMMAND ----------

from pyspark.sql.functions import expr

@feature_table
def compute_service_features(data):
  
  service_cols = ["customerID"] + [c for c in data.columns if c not in ["churnString"] + demographic_cols]
  
  #compute customer loyalty score
  cond = """case when tenure >= 24 then 4
              else case when tenure >= 12 then 2
              else 0
              end
            end +           
            case when contract = 'Two year' then 2
              else case when contract = 'One year' then 1
              else 0
              end
            end +        
            case when paymentMethod like '%automatic%' then 1
              else 0
            end  +
            case when phoneService = 'Yes' then 1
              else 0
            end +
            case when multipleLines = 'Yes' then 1
              else 0
            end +
            case when internetService != 'No' then 1
              else 0
            end"""
  
  return data.select(service_cols).withColumn("loyalty_score", expr(cond)).fillna({"TotalCharges": 0.0})

service_df = compute_service_features(telcoDF)

display(service_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Writing Feature Store Tables

# COMMAND ----------

# DBTITLE 0,Write features to the feature store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

fs.write_table(name='telco_churn.demographic_features', df = demographics_df, mode = 'merge')
fs.write_table(name='telco_churn.service_features', df = service_df, mode = 'merge')

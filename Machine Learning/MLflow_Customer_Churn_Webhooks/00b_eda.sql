-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Streamlining ML Operations Using Databricks
-- MAGIC 
-- MAGIC ##Step 0b - Exploratory Data Analysis

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Explore Data

-- COMMAND ----------

SELECT *

FROM telco_churn.bronze_customers

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Customer Distribution by Gender

-- COMMAND ----------

SELECT gender, count(*)

FROM telco_churn.bronze_customers

GROUP BY gender

-- COMMAND ----------

SELECT DISTINCT paymentMethod

FROM telco_churn.bronze_customers

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Monthly Charges by Payment Method

-- COMMAND ----------

SELECT paymentMethod as `Payment Method`, sum(monthlycharges) as `Monthly Charges`

FROM telco_churn.bronze_customers

GROUP BY paymentMethod

# Databricks notebook source
!pip install datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create Delta Table with Images and Metadata

# COMMAND ----------

# Specify the catalog and schema (database) within which to create the table
catalog_name = "field_demos"
schema_name = "luke_sandbox"
table_name = "images"

# COMMAND ----------

#load images
from datasets import load_dataset

imagenette = load_dataset(
  'frgfm/imagenette',
  'full_size',
  split='train',
  ignore_verifications=False
)

imagenette[0]["image"]

# COMMAND ----------

import io
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id

def image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Convert images to byte arrays and create Rows
rows = [Row(image=image_to_byte_array(item['image']), label=item['label']) for item in imagenette]

# Create a DataFrame from the list of Rows
df = spark.createDataFrame(rows)
# Add a unique identifier column to the DataFrame
df = df.withColumn("id", monotonically_increasing_id())

#spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
#spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# Write the DataFrame to the Delta table
df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

# Show the DataFrame
df.display()

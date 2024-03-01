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

# DBTITLE 1,Spark Image Serialization Workflow
from pyspark.sql import Row

def image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Add an index to each image and label when creating Rows
rows_with_index = [Row(id=i, image=image_to_byte_array(item['image']), label=item['label']) 
                   for i, item in enumerate(imagenette)]

# Create a DataFrame from the list of Rows with index
df_with_index = spark.createDataFrame(rows_with_index)

# Write the DataFrame with index to the Delta table
df_with_index.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

# Show the DataFrame
df_with_index.display()

# COMMAND ----------



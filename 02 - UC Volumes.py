# Databricks notebook source
# MAGIC %md
# MAGIC ##Check Image Library
# MAGIC Images can be store in volumes. This volume has 400 images of cars, motorcycles, trucks, and buses. This can be used to test with the embedding model. CLIP is capable of handling much more variety though. This can be demonstrated with the imagenette dataset from huggingface.

# COMMAND ----------

# get images from volume
images_folder = "/Volumes/field_demos/luke_sandbox/vehicle_images"

#list contents of volume
dbutils.fs.ls(images_folder+"/Car")

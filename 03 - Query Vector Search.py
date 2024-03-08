# Databricks notebook source
# MAGIC %md
# MAGIC #Query Vector Search
# MAGIC The vector database can be queried using text or an image as input.

# COMMAND ----------

!pip install torch transformers datasets pillow

# COMMAND ----------

#load model
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else \
          ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"


model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Text Query

# COMMAND ----------

prompt = "a dog playing in the snow"

#tokenize the prompt

inputs = tokenizer(prompt, return_tensors="pt")
inputs

# COMMAND ----------

text_emb = model.get_text_features(**inputs)
prompt_emb = text_emb[0].tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query with REST API
# MAGIC Here is documentation to execute the query: https://docs.databricks.com/api/workspace/vectorsearchindexes/queryindex

# COMMAND ----------

# DBTITLE 1,Semantic Vector Search Query
# Example query payload, adjust according to your needs
payload = {"query_vector": prompt_emb,
            "columns": ["index"],
            "num_results": 10
}

response = requests.post(api_url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    results = response.json()
    print("Search results:", results)
else:
    print("Failed to search the vector database:", response.text)

# COMMAND ----------

response.json()["result"]["data_array"]

# COMMAND ----------

# Assuming 'results' variable contains the response data array from vector search
search_results_df = spark.createDataFrame(response.json()["result"]["data_array"]).withColumnRenamed("_1", "index").withColumnRenamed("_2", "similarity_score")

# Load the 'images' table as a DataFrame
images_df = spark.table("field_demos.luke_sandbox.images")

# Perform the join on 'index' from search results and 'id' from images
joined_df = search_results_df.join(images_df, search_results_df.index == images_df.id)

# COMMAND ----------

joined_df.display()

# COMMAND ----------

from io import BytesIO
from PIL import Image
import IPython.display as display

# Function to convert binary data to an image and display it
def display_images(images_df):
    for row in images_df.collect():
        image_data = row['image']  # Extract binary image data
        image = Image.open(BytesIO(image_data))  # Convert to image object
        display.display(image)  # Display the image

# Assuming 'joined_df' is the DataFrame with the image column after joining
display_images(joined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Image Query

# COMMAND ----------

# MAGIC %md
# MAGIC query_index(index_name: str, columns: List[str] [, filters_json: Optional[str], num_results: Optional[int], query_text: Optional[str], query_vector: Optional[List[float]]]) → QueryVectorIndexResponse
# MAGIC Query an index.
# MAGIC
# MAGIC Query the specified vector index.
# MAGIC
# MAGIC Parameters:
# MAGIC index_name – str Name of the vector index to query.
# MAGIC
# MAGIC columns – List[str] List of column names to include in the response.
# MAGIC
# MAGIC filters_json –
# MAGIC
# MAGIC str (optional) JSON string representing query filters.
# MAGIC
# MAGIC Example filters: - {“id <”: 5}: Filter for id less than 5. - {“id >”: 5}: Filter for id greater than 5. - {“id <=”: 5}: Filter for id less than equal to 5. - {“id >=”: 5}: Filter for id greater than equal to 5. - {“id”: 5}: Filter for id equal to 5.
# MAGIC
# MAGIC num_results – int (optional) Number of results to return. Defaults to 10.
# MAGIC
# MAGIC query_text – str (optional) Query text. Required for Delta Sync Index using model endpoint.
# MAGIC
# MAGIC query_vector – List[float] (optional) Query vector. Required for Direct Vector Access Index and Delta Sync Index using self-managed vectors.
# MAGIC
# MAGIC Returns:
# MAGIC QueryVectorIndexResponse

# COMMAND ----------

!pip install databricks-vectorsearch

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

import time
index = vsc.get_index(endpoint_name="pupr_endpoint",index_name="field_demos.luke_sandbox.image_embeddings_vs")
while not index.describe().get('status')['ready']:
  print("Waiting for index to be ready...")
  time.sleep(30)
print("Index is ready!")
index.describe()

# COMMAND ----------

results = index.similarity_search(
  query_vector=prompt_emb,
  columns=["id", "image_embeddings"],
  num_results=5
  )
rows = results['result']['data_array']
for (id, text, title, score) in rows:
  if len(text) > 32:
    # trim text output for readability
    text = text[0:32] + "..."
  print(f"id: {id}  title: {title} text: '{text}' score: {score}")

# COMMAND ----------

from databricks_vectorsearch_preview import VectorSearch

# Assuming you have already created a VectorSearch object called 'vector_search'
# with the appropriate connection and index details

# Search for similar vectors
results = vector_search.search(index_name="field_demos.luke_sandbox.image_embeddings_vs",
                              vector_queries=prompt,
                              top_k=10)

for result in results:
    print(result)

# COMMAND ----------

# DBTITLE 1,Databricks Image Search Client
import requests
import json

# Replace these with the actual values
index_name = "field_demos.luke_sandbox.image_embeddings_vs"
api_url = f"https://e2-dogfood.staging.cloud.databricks.com/api/2.0/vector-search/indexes/{index_name}/query"
token = "your_databricks_token"

# Example query payload, adjust according to your needs
payload = {"query_vector": prompt_emb,
            "columns": ["index"],
            "num_results": 10
}

headers = {
    "Authorization": f"Bearer xxxxxx",
    "Content-Type": "application/json"
}

# COMMAND ----------



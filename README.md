# Multimodal Vector Search

[Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) supports multimodal inputs through the use of third party embedding models. In this project, we will demo OpenAI's CLIP model independently. Then we will demonstrate how it can be used with Databricks Vector Search product.

###00 - CLIP Embedding Demo
Load public images from a Hugging Face dataset and to be used with OpenAI's open source CLIP embedding model. The purpose of this notebook is to showcase how the model can be used to create embeddings for both text and images.

###01 - Initialize Data
Load image data to a Volume in Unity Catalog. The images in this catalog will later be converted to embeddings and stored in a vector search index. Images that are added to this volume later on can be automatically processed through our pipeline to be included in our index.

###02 - Create Vector Search Table
Convert the images from our volume in Unity Catalog to embeddings. The embeddings will be stored in a delta table that can be converted to a vector index through the UI or with the Databricks SDK.

###03 - Query Vector Search
Query the vector index for similar content with text or an image as input. Images from the vector index will be returned based on inferred similarity scores.
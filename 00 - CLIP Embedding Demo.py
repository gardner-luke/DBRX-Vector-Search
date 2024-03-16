# Databricks notebook source
# MAGIC %md
# MAGIC #CLIP Embedding Demo
# MAGIC Create embeddings for a text prompt and images. Store the images in a vector index and query them using the text prompt.
# MAGIC
# MAGIC 1. Create embedding for text prompt
# MAGIC 2. Create embedding for image
# MAGIC 3. Create embeddings for several images and load to vector search index.
# MAGIC 4. Use embedding from text prompt to query vector index.
# MAGIC
# MAGIC ###OpenAI CLIP Documentation and Tutorials
# MAGIC [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) | [Medium Article](https://towardsdatascience.com/quick-fire-guide-to-multi-modal-ml-with-openais-clip-2dad7e398ac0) | [YouTube Tutorial](https://www.youtube.com/watch?v=989aKUVBfbk)

# COMMAND ----------

!pip install torch transformers datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Images
# MAGIC These images come from a public dataset on [Hugging Face](https://huggingface.co/). The dataset [Imagenette](https://huggingface.co/datasets/frgfm/imagenette) contains nearly 10,000 images that can be classified into 10 classes (labels).

# COMMAND ----------

# load images
from datasets import load_dataset

imagenette = load_dataset(
  'frgfm/imagenette',
  'full_size',
  split='train',
  ignore_verifications=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Model
# MAGIC [OpenAI's CLIP](https://huggingface.co/openai/clip-vit-base-patch32) embedding model is built specifically for computer vision tasks and allows us to create numberic representations of images and text alike. This enables us to query images and/or text with either mode as input.

# COMMAND ----------

# load model
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
# MAGIC ##Create Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ###Text Embeddings
# MAGIC The prompt defined in the first line can be anything you'd like. In this example we will use "a dog in the snow". This can be updated to whatever you'd like. Results will vary depending on the input prompt.

# COMMAND ----------

prompt = "a dog in the snow"

# tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")
inputs

# COMMAND ----------

# use the tokenized prompt to get 512 dimension embedding stored in the variable text_emb
text_emb = model.get_text_features(**inputs)
text_emb.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###Image Embeddings
# MAGIC Similarly to the text embedding above we will use the model to generate a 512 dimension embedding of the image. Instead of pre-processing the data with the tokenizer, we will use the processor.

# COMMAND ----------

# resizing the image with proceessor
# expected shape is torch.Size([1, 3, 224, 224])
image = processor(text=None,
                  images = imagenette[0]['image'],
                  return_tensors="pt")['pixel_values'].to(device)
image.shape

# COMMAND ----------

# after this line you will have a 512 dimension embedding vector
image_emb = model.get_image_features(image)
image_emb.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###Batch Processing
# MAGIC Using a subset of 100 images, we will create embeddings of 16 images at a time and store the results in a variable called image_arr.

# COMMAND ----------

# get a subset of 100 images for this experiment
import numpy as np

np.random.seed(0)
sample_idx = np.random.randint(0, len(imagenette)+1,100).tolist()
images = [imagenette[i]['image'] for i in sample_idx]
len(images)

# COMMAND ----------

from tqdm.auto import tqdm

batch_size = 16
image_arr = None

for i in tqdm(range(0, len(images), batch_size)):
  # select batch of images
  batch = images[i:i+batch_size]
  # process and resize images
  batch = processor(text=None,
                  images = batch,
                  return_tensors="pt",
                  padding=True,
                  is_train=False)['pixel_values'].to(device)
  
  # get image embeddings
  batch_emb = model.get_image_features(pixel_values=batch)
  # convert to numpy array
  batch_emb = batch_emb.squeeze(0)
  batch_emb = batch_emb.cpu().detach().numpy()
  # add to larger array of all image embeddings
  if image_arr is None:
    image_arr = batch_emb
  else:
    image_arr = np.concatenate((image_arr, batch_emb), axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculate Scores
# MAGIC Scores are stored in an image array and normalized. The normalized data is then compared to the embedding from our original prompt.

# COMMAND ----------

# normalize the values in the image array
image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)

# get the text embedding from ealier
text_emb = text_emb.cpu().detach().numpy()

# calculate the scores between the text embedding and the image array
scores = np.dot(text_emb, image_arr)

# return only the top 5
top_k = 5
idx = np.argsort(-scores[0])[:top_k]

idx

# COMMAND ----------

# MAGIC %md
# MAGIC ###Show Results
# MAGIC The scores have been normalized and are presented above each image in the output of the final code block. A larger number indicates a closer relationship between the input data and the result.

# COMMAND ----------

import matplotlib.pyplot as plt

# show the images and their scores
for i in idx:
  print(scores[0][i])
  plt.imshow(images[i])
  plt.show()

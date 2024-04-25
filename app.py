import random
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import nltk
from nltk.stem import PorterStemmer
import json
import os
from torch.utils.data import DataLoader

nltk.download('punkt')

# Function to train the model
def train_model(intents_file, data_file):
  """
  This function trains a neural network model for the chatbot based on the provided intents file and saves the trained model data.

  Args:
      intents_file (str): Path to the JSON file containing chatbot intents data (patterns and tags).
      data_file (str): Path to the file where the trained model data will be saved.
  """

  # Load intents data from JSON file
  with open(intents_file, 'r') as f:
    intents = json.load(f)

  all_words = []
  tags = []
  xy = []
  # Process intents data for training
  for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
      w = tokenize(pattern)
      all_words.extend(w)
      xy.append((w, tag))

  # Stem and lower each word
  ignore_words = ['?', '.', '!']
  all_words = [stem(w) for w in all_words if w not in ignore_words]
  # Remove duplicates and sort
  all_words = sorted(set(all_words))
  tags = sorted(set(tags))

  print(len(xy), "patterns")
  print(len(tags), "tags:", tags)
  print(len(all_words), "unique stemmed words:", all_words)

  # Hyper-parameters (consider tuning these for better performance)
  num_epochs = 1000
  batch_size = 8
  learning_rate = 0.001
  input_size = len(all_words)
  hidden_size = 8
  output_size = len(tags)
  print(input_size, output_size)

  # Create the neural network model
  model = NeuralNet(input_size, hidden_size, output_size)

  # Define loss function and optimizer (consider experimenting with different options)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train the model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  for epoch in range(num_epochs):
    # Load data from your intents.json file (assuming it's in the same directory)
    with open(intents_file, 'r') as f:
      intents = json.load(f)

    xy = []
    # Process intents data again to create training data (words and labels)
    for intent in intents['intents']:
      tag = intent['tag']
      for pattern in intent['patterns']:
        w = tokenize(pattern)
        xy.append((w, tag))

    # Convert words to bag-of-words and create tensors
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
      bag = bag_of_words(pattern_sentence, all_words)
      X_train.append(bag)
      label = tags.index(tag)
      y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)

    # Training loop
    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)  # Corrected DataLoader usage
    for (words, labels) in train_loader:  # Iterate over batches
      words = words.to(device)
      labels = labels.to(dtype=torch.long).to(device)

      # Forward pass
      outputs = model(words)
      loss = criterion(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print training progress (optional)
      if (epoch + 1) % 100 == 0:
        print('Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  # Save the trained model data
  data = {
      "model_state": model.state_dict(),
      "input_size": input_size,
      "hidden_size": hidden_size,
      "output_size": output_size,
      "all_words": all_words,
      "tags": tags
  }
  torch.save(data, data_file)
  print(f'training complete. file saved to {data_file}')

# Define functions for preprocessing and prediction
def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  stemmer = PorterStemmer()
  return stemmer.stem(word.lower())  # Lowercase the word before stemming

def bag_of_words(tokenized_sentence, words):
  """
  Creates a bag-of-words representation for a given sentence.

  Args:
      tokenized_sentence (list): List of words from the sentence after tokenization.
      words (list): List of all unique words considered for the bag.

  Returns:
      numpy.ndarray: A numpy array representing the bag-of-words for the sentence.
  """
  sentence_words = [stem(word) for word in tokenized_sentence]
  bag = np.zeros(len(words), dtype=np.float32)
  for idx, w in enumerate(words):
    if w in sentence_words:
      bag[idx] = 1
  return bag

# Define the neural network model
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    return out  # No activation or softmax at the end for CrossEntropyLoss

# Load data or train if data file doesn't exist
data_file = "data.pth"
if not os.path.isfile(data_file):
  # Train the model if data file is missing
  train_model("intents.json", data_file)  # Assuming intents.json is in the same directory

# Load data from saved file
data = torch.load(data_file)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "PsyBot"

# Streamlit app
def main():
  st.title("Mental Health ChatBot")

  user_input = st.text_input("You:")

  if user_input == "quit":
    st.stop()

  if user_input:
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
      # Load intents from your JSON file
      with open("intents.json", "r") as f:
        intents = json.load(f)

      for intent in intents['intents']:
        if tag == intent["tag"]:
          st.write(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
      st.write(f"{bot_name}: I am still learning. Perhaps I can't understand you at this moment. You can find resources for mental health support here: [National Alliance on Mental Illness (NAMI)](https://www.nami.org/Home)")  # Include a helpful resource

if __name__ == '__main__':
  main()

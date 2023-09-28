from __future__ import unicode_literals, print_function
import random
import spacy
import os
import json
from spacy.training.example import Example
import sys
from pathlib import Path
# # Load the pre-trained spaCy model
# !python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")


train_data =  str(sys.argv[1])
test_data = str(sys.argv[2])
#if no arguments are given, use the default as above
if len(sys.argv) == 1:
    train_data = "6_3.json"
    test_data = "06_9.json"
# Load the training data from the JSON file
with open("datasets/"+train_data, "r") as file:
    TRAIN_DATA = json.load(file)
with open("datasets/"+test_data, "r") as file:
    TEST_DATA = json.load(file)

model = None
output_dir = 'output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_iter = 100

# Create a blank spaCy model or load an existing one
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

# Set up the pipeline for Named Entity Recognition (NER)
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')

# Add entity labels to the NER pipeline
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

# Disable other pipeline components for training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

# Training loop
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update(
                [text],
                [annotations],
                drop=0.5,
                sgd=optimizer,
                losses=losses)
        print(losses)

# Save the trained model to a directory
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

# Test the trained model
for text, _ in TEST_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

# Load the model
model = spacy.load('model_name')

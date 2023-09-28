from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training.example import Example
import json
import sys
# Load the spaCy model
nlp1 = spacy.load("en_core_web_sm")

# Train Data
TRAIN_DATA = []


#if no arguments are given, use the default as above
if len(sys.argv) == 1:
    train_data = "6_3.json"
    test_data = "06_9.json"
else:
    train_data = str(sys.argv[1])
    test_data = str(sys.argv[2])

# Read the JSON data line by line
with open("datasets/"+train_data, 'r') as json_file:
    for line in json_file:
        record = json.loads(line)
        text = record["text"]
        entities = []

        # Extract entity information
        for entity_info in record["entities"]:
            start, end, label = entity_info
            entities.append((start, end, label))

        # Append to TRAIN_DATA
        TRAIN_DATA.append((text, {"entities": entities}))

# Define variables
model = None
output_dir = 'output'
n_iter = 100

# Load the model
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

# Set up the pipeline
# Check if 'ner' component already exists in the pipeline
if 'ner' not in nlp.pipe_names:
    # Create a new NER component
    ner = nlp.add_pipe('ner')
    # nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')

# Train the Recognizer
# Add labels to the NER component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)

# Test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# Save the model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

# Test the saved model
# Initialize the TEST_DATA list
TEST_DATA = []

# Read the JSON data line by line
with open("datasets/"+test_data, 'r') as json_file:
    for line in json_file:
        record = json.loads(line)
        text = record["text"]
        entities = []

        # Extract entity information
        for entity_info in record["entities"]:
            start, end, label = entity_info
            entities.append((start, end, label))

        # Append to TEST_DATA
        TEST_DATA.append((text, {"entities": entities}))

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text, _ in TEST_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

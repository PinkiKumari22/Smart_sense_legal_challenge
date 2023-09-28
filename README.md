# Legal Entity Recognition with spaCy

## Overview

This project demonstrates how to train a Named Entity Recognition (NER) model using spaCy to extract legal entities from legal documents. It includes a dataset in JSON format and annotations in the IOB (Inside, Outside, Beginning) tagging scheme. The trained model can recognize entities such as organizations, persons, dates, and judges in legal texts.

## Data Formats

### JSON Data Format

The training data is provided in the following JSON format:

```json
{
    "doc_id": 1835,
    "text": "1 The applicants Sharman Networks Ltd ('Sharman Networks'), Sharman License Holdings Ltd ('Sharman License') and Ms Nicola Anne Hemming ('Ms Hemming') are each the subject of asset preservation orders made by Wilcox J on 22 March 2005 ('the Mareva orders').",
    "entities": [
        [17, 37, "Organization"],
        [38, 58, "Organization"],
        [60, 88, "Organization"],
        [89, 108, "Organization"],
        [113, 135, "person"],
        [136, 150, "person"],
        [221, 234, "Date"],
        [209, 217, "Judges"]
    ],
    "username": "admin"
}
```

### Annotation Format (IOB Tagging)

The annotation data is provided in the IOB tagging scheme in the legal_train.text file:
```
-DOCSTART- -X- -X- O
1 Others 
The Others
applicants Others
Sharman B-Organization
Networks I-Organization
Ltd L-Organization
...
22 B-Date
March I-Date
2005 L-Date
the Others
Mareva Others
```

# Running the Script

To execute the script, follow these steps in your terminal or command prompt:

1. Download the spaCy English model:

   ```bash
   python3 -m spacy download en_core_web_sm
   ```
2. Run the script "challenge.py" :
   Running the Script with Command-Line Arguments
   You can run the script "challenge.py" with custom training and test data files using command-line arguments. By default, the script uses "6_3.json" as the training data and "06_9.json" as the test data. To specify custom data files, follow the format:

```bash
python3 challenge.py <training_data.json> <test_data.json>
```
Default setting are: training json is **6_3.json**  snd test json is **06_9.json** and can run as
   ```bash
   python3 challenge.py
   ```




## Jupyter Notebooks:
1. the notebook _challenge.ipynb_ has teh demo code to train and test a trained model on the given dataset
2. other notebook for EDA

# Solution Approach
We use **spaCy** library to train on the given dataset.

1. Load and preprocess the training data from the JSON file.
2. Create or load a spaCy model and configure it for NER.
3. Prepare training examples in spaCy format using the provided annotations.
4. Train the NER model on the training data with multiple iterations.
5. Save the trained model to a directory for future use.
6. Test the trained model on sample text to extract legal entities.

## Libraries Used

- **spaCy:** A popular NLP library for building NER models.
- **Python:** The programming language used for the project.

## Getting Started

1. Install spaCy: `pip install spacy`
2. Install spaCy English model: `python -m spacy download en_core_web_sm`
3. Clone this GitHub repository: `git clone <repository_url>`
4. Navigate to the project directory: `cd legal-entity-recognition`

## Usage

1. Prepare your training data in the JSON format as shown in the example.
2. Execute the provided Jupyter Notebook to train and test the spaCy NER model.

## Acknowledgments

- [spaCy](https://spacy.io/)
- Example data provided by Mahima Singh.


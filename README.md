# Fine-Tuning and Evaluation Tutorial for MLADS 2025

**Accelerating LLM and SLM with Finetuning: A Coding Tutorial for Evaluating Accuracy, Latency, and Cost to Find the Best Model**  
*by Daniel Taube*

## Overview

This tutorial demonstrates how to fine-tune sequence classification models on the IMDB dataset using the Hugging Face Transformers library. You’ll learn how to:

- Prepare and sample data from a large-scale IMDB dataset.
- Fine-tune a Small Language Model (SLM) like `distilbert-base-uncased` for sentiment analysis.
- Extend the process to fine-tune larger models (LLMs) such as `bert-large-uncased`.
- Evaluate the models based on accuracy, latency, and inferential cost.

This live coding session was designed for MLADS 2025.

## Requirements

- **Python 3.11** (or later)
- **PyTorch**
- **Hugging Face Transformers**
- **Datasets**
- **Evaluate**
- **Pandas**, **NumPy**, **tqdm**

### Install Dependencies

```bash
pip install -r requiments.txt
```

Data Preparation
----------------
Loading the Data:
The notebook loads the IMDB dataset (stored in Parquet files) and creates three splits (train, test, and unsupervised).

Sampling:
A subset of 1000 positive and 1000 negative reviews is randomly sampled from the training and test sets. The resulting dataframes are saved as imdb_train.csv and imdb_test.csv.

Fine-Tuning the SLM
-------------------
Tokenizer and Model:
The tutorial uses the DistilBertTokenizer and DistilBertForSequenceClassification (a ~66M parameter model) for fine-tuning.

Tokenization:
A function is defined to tokenize the review text with truncation and padding.

Training:
The Hugging Face Trainer is configured with training arguments (learning rate, batch size, number of epochs, etc.) and an accuracy metric.

Evaluation:
The trained model is evaluated on the test set and compared against sample inputs.

Fine-Tuning the LLM
-------------------
The tutorial includes a function (fine_tune_slm) that is designed to work with larger models such as bert-large-uncased (approximately 340M parameters).

The process remains largely the same as for the SLM, highlighting how to easily swap model architectures for different parameter scales.

Evaluating Accuracy and Latency
-------------------------------
Evaluation Functions:
The notebook defines functions (evaluate_model_accuracy_latency and evaluate_all_models) that:
- Process a subset of test data.
- Compute the model's accuracy.
- Measure the total latency during inference.

Results:
A comparison table is generated to show how the SLM and LLM compare in terms of:
- Accuracy (e.g., SLM vs. fine-tuned SLM).
- Latency (how quickly each model processes input).
- Indicative cost implications based on resource usage.

How to Run the Tutorial
-----------------------
Clone or Download:
Get a copy of the notebook file.

Install Dependencies:
Make sure all required Python packages are installed.

Run the Notebook:
Execute the notebook cells in order:
- Data loading and sampling.
- Fine-tuning for both SLM and LLM.
- Evaluation of accuracy and latency.

Experiment:
You can modify hyperparameters or swap model architectures to see how performance changes.

Notes
-----
Data Sampling:
The tutorial uses a small subset of the IMDB data (2000 examples) for faster training and evaluation. For more robust results, consider using a larger portion of the dataset.

Resource Considerations:
Larger models may require GPU acceleration. The notebook is configured to use CPU by default if a GPU is not available.

Prompting:
The evaluation section demonstrates the effect of adding a prompt to the input text, affecting both accuracy and latency.


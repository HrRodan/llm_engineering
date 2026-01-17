# Pricer: Product Price Prediction

This directory contains a complete pipeline for predicting product prices based on their textual descriptions. It leverages Large Language Models (LLMs) to summarize and standardize noisy Amazon product data, and then uses a Deep Neural Network (DNN) to perform regression on the processed text embeddings.

## Project Structure

### 1. Data Structure & Loading
- **`items.py`**: Defines the core data model `Item` using Pydantic. Handles data validation and HuggingFace Hub integration.
- **`loaders.py`**: Handles downloading and parallel processing of the "Amazon Reviews 2023" dataset. Uses multiprocessing to efficiently load and parse large chunks of data.

### 2. Data Processing
- **`parser.py`**: A robust parsing module that cleans raw HTML/JSON input. It removes irrelevant metadata (like part numbers), normalizes units (e.g., weights), and ensures data quality before it enters the pipeline.
- **`preprocessor.py`**: A synchronous wrapper for LLM interactions. It uses `litellm` (and Groq/OpenAI models) to rewrite and summarize verbose product descriptions into a concise, standardized format (Title, Category, Brand, Description, Features).

### 3. Batch Operations
- **`batch.py`**: specialized module for handling large-scale data processing using the Groq Batch API. It manages the lifecycle of creating JSONL batch files, uploading them, submitting jobs, and retrieving/applying results, allowing for cost-effective processing of thousands of items.

### 4. Neural Network Model
- **`deep_neural_network.py`**: Implements the price prediction model using **PyTorch**.
  - **Architecture**: A Residual Neural Network (ResNet) adapted for regression, featuring multiple `ResidualBlock` layers with LayerNorm and Dropout to prevent overfitting.
  - **Training**: Includes a complete `DeepNeuralNetworkRunner` that handles vectorization (`HashingVectorizer`), data normalization (log-price), and the training loop with `AdamW` optimizer and Cosine Annealing learning rate schedule.

### 5. Evaluation & Visualization
- **`evaluator.py`**: A comprehensive evaluation suite using **Plotly**.
  - Generates interactive scatter plots comparing Predicted vs. Actual prices.
  - Calculates key metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
  - Visualizes error trends and confidence intervals to analyze model stability.

## Workflow

1.  **Ingestion**: `ItemLoader` pulls raw data from the Amazon dataset.
2.  **Cleaning**: `Parser` scrubs the data and converts it into `Item` objects.
3.  **Preprocessing**: `Batch` or `Preprocessor` uses an LLM to generate high-quality summaries from the noisy raw text.
4.  **Vectorization**: The summaries are converted into numerical vectors using a hashing vectorizer.
5.  **Training**: The `DeepNeuralNetwork` is trained on these vectors to predict the log-normalized price.
6.  **Inference & Eval**: The `Evaluator` runs the model on a test set and produces visual reports on its accuracy.

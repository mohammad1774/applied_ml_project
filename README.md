# Root Notebooks Overview
## 1. Bag_of_words_features.ipynb

Classical NLP using Bag-of-Words (BoW)

This notebook implements the most fundamental text representation technique: Bag of Words. Text documents are converted into sparse count vectors after stopword removal.

Key components:

- Feature extraction using CountVectorizer

- Models evaluated:

  - Logistic Regression

  - Random Forest

  - XGBoost

  - MLP (Keras-based)

- Metrics:

  - Accuracy

  - Weighted F1-score

  - Training time

  - Inference time

  - Analytical FLOPs per sample

Purpose:
Serves as the baseline model to understand how linear and tree-based classifiers behave with sparse, high-dimensional representations.
---

## 2. tf_idf_vectorizer.ipynb

Statistical weighting with TF–IDF

This notebook extends classical NLP by replacing raw word counts with TF–IDF (Term Frequency–Inverse Document Frequency) features, capturing word importance across the corpus.

Key components:

- Feature extraction using TfidfVectorizer (unigrams + bigrams)

- Models evaluated:

  - Logistic Regression

  - Random Forest

  - XGBoost

  - MLP (Keras-based)

- Performance and efficiency metrics identical to BoW

- FLOPs estimation based on feature dimensionality

Purpose:
Demonstrates how statistical reweighting improves generalization while retaining classical ML interpretability and low computational cost.
---

## 3. embedding_deep_learning.ipynb

Learned word embeddings with deep neural networks

This notebook transitions from feature engineering to learned representations using trainable embeddings.

Key components:

- Keras Tokenizer + sequence padding

- Embedding matrix learned via backpropagation

- Models implemented:

  - CNN (Conv1D + Global Max Pooling)

  - MLP (Dense layers)

- Analysis of:

  - Embedding updates

  - Training vs inference cost

  - Representation learning behavior

Purpose:
Illustrates how neural models learn semantic structure directly from data, enabling richer representations than sparse vectors.
---

## 5. bert_transformer_code.ipynb

End-to-end BERT fine-tuning (Transformer model)

This is the most advanced notebook in the repository and represents the final stage of the architectural progression.

Key components:

- Full fine-tuning of bert-base-uncased

- Tokenization with attention masks

- TensorFlow + Hugging Face integration

- Multi-GPU training using MirroredStrategy

- Custom classification head on the [CLS] token

- Evaluation of:

  - Accuracy and F1-score

  - Training and inference cost

  - System-level considerations (GPU utilization)

Purpose:
Demonstrates modern NLP systems used in production, emphasizing computational cost, scalability, and hardware dependence.
---

## Conceptual Progression

The root notebooks are intentionally ordered to reflect an engineering-first learning path:

1. Bag of Words → sparse, interpretable, fast

2. TF–IDF → statistically informed features

3. Trainable Embeddings → learned semantics

4. Fine-tuned BERT → state-of-the-art Transformer model

This progression enables a clear comparison of accuracy vs efficiency trade-offs, which is essential for real-world deployment decisions.
---

## Dataset

- AG News Classification Dataset

- 4 classes: World, Sports, Business, Sci/Tech

- 120,000 training samples, 7,600 test samples
---

## Course Context

Course: ELG 5255 – Applied Machine Learning
Focus: Model architecture, computational efficiency, and system-level reasoning
Author: Mohammad (300480272)

# PyTorch Deep Learning Practice

A hands-on collection of PyTorch notebooks covering neural-network fundamentals, computer vision, GPU training, hyperparameter optimization, transfer learning, and sequence models.

This repository documents my progression from building basic `nn.Module` models to training ANN/CNN architectures, tuning models with Optuna, using pretrained VGG16 features, and experimenting with RNN/LSTM models for NLP tasks.

## What this repository covers

- PyTorch tensors, `nn.Module`, forward passes, loss functions, and optimizers
- Custom `Dataset` and `DataLoader` pipelines
- Artificial neural networks for Fashion-MNIST classification
- GPU-aware training workflows
- Batch normalization and dropout
- Convolutional neural networks
- Hyperparameter optimization with Optuna
- Transfer learning with pretrained VGG16
- RNN-based question answering
- LSTM-based next-word prediction
- A complete preprocessing-to-training classification workflow

## Notebook guide

| Notebook | Focus |
|---|---|
| `pytorch_nn_module.ipynb` | PyTorch `nn.Module` and model architecture fundamentals |
| `pytorch_training_pipeline.ipynb` | End-to-end binary classification pipeline using the Wisconsin breast-cancer dataset, including train/test splitting, scaling, encoding, training, and evaluation |
| `ann_fashion_mnist_pytorch.ipynb` | Baseline fully connected ANN for Fashion-MNIST |
| `ann_fashion_mnist_pytorch_gpu.ipynb` | ANN training with GPU support |
| `ann_fashion_mnist_pytorch_gpu_optimized.ipynb` | ANN with batch normalization, dropout, and other training improvements |
| `ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb` | Optuna-based tuning of ANN depth, width, dropout, optimizer, batch size, learning rate, weight decay, and epochs |
| `cnn_fashion_mnist_pytorch_gpu.ipynb` | CNN-based Fashion-MNIST classifier using convolution, batch normalization, dropout, and GPU training |
| `cnn_optuna.ipynb` | Optuna search over CNN architecture and training hyperparameters |
| `transfer_learning_fashion_mnist_pytorch_gpu.ipynb` | Transfer learning for Fashion-MNIST using a pretrained VGG16 backbone and a custom classifier |
| `pytorch_rnn_based_qa_system.ipynb` | Simple RNN question-answering experiment using tokenization, vocabulary construction, embeddings, and recurrent modeling |
| `pytorch_lstm_next_word_predictor.ipynb` | LSTM next-word prediction with custom sequence data preparation and training |

## Learning progression

```text
PyTorch fundamentals
        ↓
Custom Dataset + DataLoader
        ↓
ANN baseline
        ↓
GPU training
        ↓
BatchNorm + Dropout
        ↓
Optuna hyperparameter tuning
        ↓
CNNs
        ↓
Transfer learning
        ↓
RNN / LSTM sequence models
```

## Selected experiments

### Fashion-MNIST ANN

The ANN notebooks progressively move from a basic fully connected network to GPU training, regularization, normalization, and automated hyperparameter search.

### Fashion-MNIST CNN

The CNN experiments introduce convolutional layers and model tuning. The stored Optuna run in `cnn_optuna.ipynb` reports a best validation accuracy of approximately **92.23%**.

### Transfer learning

`transfer_learning_fashion_mnist_pytorch_gpu.ipynb` uses a pretrained **VGG16** model and replaces its classifier with a task-specific network for the 10 Fashion-MNIST classes.

### Sequence modeling

The NLP notebooks explore two recurrent architectures:

- a simple RNN for a small question-answering dataset
- an LSTM for next-word prediction

These notebooks are learning experiments intended to demonstrate sequence preparation, embeddings, recurrent hidden states, training loops, and inference.

## Tech stack

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Optuna
- NLTK
- torchinfo
- Pillow
- Google Colab / Jupyter Notebook

## Running the notebooks

Clone the repository:

```bash
git clone https://github.com/htoor2026/pytorch.git
cd pytorch
```

Install the common dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib optuna nltk torchinfo pillow
```

Then open the notebooks with Jupyter:

```bash
jupyter notebook
```

You can also upload any notebook directly to Google Colab.

## Dataset notes

The repository includes:

```text
100_Unique_QA_Dataset.csv
```

Several Fashion-MNIST notebooks currently reference files such as:

```text
fashion-mnist_train.csv
fmnist_small.csv
```

Those files are not stored in this repository, so they must be downloaded or uploaded separately before running the corresponding notebooks.

The breast-cancer training pipeline reads its dataset from a public GitHub raw-data URL.

## Repository purpose

This is a **deep-learning learning repository**, not a single production application. The emphasis is on understanding PyTorch by implementing models, training loops, optimization strategies, and experiments directly in notebooks.

The progression is intentionally incremental: later notebooks build on ideas introduced in earlier ones.

## Next improvements

- Organize notebooks into `fundamentals/`, `computer-vision/`, and `nlp/`
- Add a reproducible `requirements.txt`
- Standardize dataset loading so notebooks run without manual path changes
- Add consistent train/validation/test evaluation
- Track experiments and model configurations more systematically
- Add confusion matrices and per-class metrics for classification experiments
- Convert reusable training logic into Python modules

---

**Author:** [Harkamal Toor](https://github.com/htoor2026)

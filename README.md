# Transformer Model from Scratch

This repository contains an implementation of the Transformer model from scratch using PyTorch. The Transformer model is a powerful architecture for sequence-to-sequence tasks, such as machine translation, text summarization, and language modeling.

## Architecture

The Transformer model is based on the attention mechanism, which allows the model to focus on the most relevant parts of the input sequence when generating the output sequence. The architecture consists of an encoder and a decoder, both composed of multiple layers of multi-head attention and feed-forward networks.

<!-- ![Transformer Architecture](transformer_architecture.webp) -->
<p align="center">
  <img src="transformer_architecture.webp" width="400" height="600">
</p>

## Project Structure

This project includes the following files:

- `config.py`: Defines the model parameters such as batch size, learning rate, model dimensions, and variables for TensorBoard.
- `dataset.py`: Defines the data processing pipeline. It uses the `datasets` library to load the OPUS books dataset for the language translation task.
- `model.py`: Contains the implementation of all the Transformer layers, including the multi-head attention, feed-forward network, and the complete Transformer model.
- `model_tokenize.py`: Builds the tokenizer using the Hugging Face tokenizer library for word-level tokenization.
- `train.py`: The main file that initializes everything and trains the model. It also includes greedy decoding for inference.
- `utils.py`: Contains utility functions for getting the data loader and initializing the model based on the provided configuration.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/transformer-from-scratch.git
    ```
2. Install the required dependencies::

    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:

    ```bash
    python3 train.py
    ```

## Resources

During the development of this project, the following resources were consulted:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper by Vaswani et al.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - A helpful blog post that explains the Transformer architecture in detail.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - The official documentation for PyTorch.
- [Umar Jamil's Youtube Channel](https://www.youtube.com/@umarjamilai/videos)- The best youtube for understanding and implementing the models. 

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.



# Playing Card Classification 🃏

Welcome to my project on playing card classification! The idea behind this project is simple yet exciting: to teach a deep learning model how to recognize and classify playing cards. Whether it’s the Ace of Spades or the King of Hearts, the model can tell them apart with confidence. 

This project showcases the power of transfer learning using a custom dataset and the EfficientNet-B0 architecture, along with some cool visualization features.

---

## What's This All About?

Ever wondered how machines "see" and understand the world? With this project, I explore how to train a neural network to recognize 53 types of playing cards, including all suits and ranks, plus the Joker. It’s a hands-on experiment in computer vision and transfer learning!

---

## Highlights of the Project

- **Custom Dataset Handling**: I used PyTorch's `ImageFolder` for dataset management, complete with image transformations.
- **Transfer Learning**: Leveraged the pre-trained EfficientNet-B0 model to build a classifier without training from scratch.
- **Prediction Visualization**: The project provides easy-to-interpret visualizations of model predictions.
- **Beginner-Friendly**: If you're just getting started with PyTorch, this project is a great introduction to deep learning concepts.

---

## The Dataset

The dataset consists of playing card images organized into folders by class. Each class represents a unique combination of card rank and suit. The dataset is divided into three parts: 

- **Training Set**: Used to train the model.
- **Validation Set**: Helps evaluate the model during training.
- **Test Set**: Used to measure final model performance.

Here’s what the structure looks like:
```
/dataset_cards/
    /train/
        /ace of spades/
        /king of hearts/
        ...
    /valid/
    /test/
```

---

## The Model

The project uses **EfficientNet-B0**, a state-of-the-art convolutional neural network that is both powerful and efficient. I customized it by adding a classifier layer with 53 outputs to match the number of classes (card types).

---

## How It Works

### 1. Setting Up
- Load and preprocess the dataset with transformations like resizing and normalization.
- Create PyTorch `DataLoader` objects to handle batch processing during training.

### 2. Training
The model is trained using a simple training loop with cross-entropy loss and Adam optimizer. Each epoch includes a training phase and a validation phase to monitor performance.

### 3. Predicting and Visualizing
Want to see the magic in action? After training, the model can classify any playing card image. Predictions are visualized with:
- **The Original Image**: So you know what the model is looking at.
- **Probability Bar Chart**: Displays the likelihood of each class.

---

## What You'll Need

- **Libraries**:
    - `torch`, `torchvision`, `timm`
    - `matplotlib`, `pandas`, `numpy`
    - `tqdm` for progress tracking

- **Environment**: Google Colab is highly recommended, especially for using a GPU.

---

## How to Run This

1. Clone the repository and set up your environment:
    ```bash
    git clone https://github.com/yourusername/playing-card-classification.git
    cd playing-card-classification
    pip install -r requirements.txt
    ```

2. Mount your Google Drive if using Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. Organize your dataset as described earlier.

4. Train the model and evaluate it:
    ```python
    python train.py
    ```

5. Run the visualization script to see predictions:
    ```python
    python visualize.py
    ```

---

## Cool Visuals 🎨

Here’s an example of how predictions are visualized:
- Left: The input image (e.g., Ace of Diamonds).
- Right: A bar chart showing prediction probabilities for each class.

The bar chart makes it easy to see how confident the model is about its predictions!

---


Thanks for checking out my project! I hope you enjoy exploring the world of computer vision through this fun and engaging use case. Feel free to reach out if you have questions or suggestions. 😊

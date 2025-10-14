import argparse
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for Chinese character display
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# Task 1: Data Loading and Preprocessing
# ============================================================================


def load_and_explore_data() -> (
    tuple[NDArray[np.float64], NDArray[np.float64], list[str], list[str]]
):
    """
    Load the Iris dataset and perform exploratory data analysis.

    Returns
    -------
    X : NDArray[np.float64] of shape (150, 4)
        Feature matrix containing 150 samples with 4 features each.
    y : NDArray[np.float64] of shape (150,)
        Target labels for the 150 samples (0, 1, or 2).
    feature_names : list of str
        Names of the 4 features (sepal length, sepal width, petal length, petal width).
    target_names : list of str
        Names of the 3 iris species (setosa, versicolor, virginica).

    Notes
    -----
    This function loads the Iris dataset from sklearn and displays comprehensive
    statistical information including shape, missing values, and class distribution.
    """
    # Load Iris dataset from sklearn
    iris: Bunch = load_iris()  # type: ignore
    X: NDArray[np.float64] = iris.data
    y: NDArray[np.float64] = iris.target
    feature_names: list[str] = iris.feature_names
    target_names: list[str] = iris.target_names.tolist()

    # Convert to DataFrame for easier exploration
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = y

    print("=" * 80)
    print("IRIS DATASET EXPLORATION")
    print("=" * 80)

    # Display basic information
    print(f"\n数据集形状: {X.shape}")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别数量: {len(target_names)}")
    print(f"类别名称: {target_names}")

    # Check for missing values 检查是否存在缺失值或异常值
    print(f"\n缺失值: {df.isnull().sum().sum()}")

    # Statistical summary
    print("\n各项统计学指标总结:")
    print(df.describe())

    # Class distribution
    print("\n类别的分布:")
    for i, name in enumerate(target_names):
        count: int = np.sum(y == i)
        print(f"  {name}: {count} samples")
    return X, y, feature_names, target_names


def preprocess_data(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray, StandardScaler]:
    """
    Preprocess the data with standardization and train-test split.

    Parameters
    ----------
    X : NDArray[np.float64] of shape (n_samples, n_features)
        Feature matrix to be split and standardized.
    y : NDArray[np.float64] of shape (n_samples,)
        Target labels corresponding to X.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility of the train-test split.

    Returns
    -------
    X_train_scaled : NDArray of shape (n_train_samples, n_features)
        Standardized training feature matrix.
    X_test_scaled : NDArray of shape (n_test_samples, n_features)
        Standardized test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    scaler : StandardScaler
        Fitted StandardScaler object for potential future transformations.

    Notes
    -----
    Feature standardization is performed using Z-score normalization:
    X_scaled = (X - mean) / std
    This is crucial for KNN as it's sensitive to feature scales.
    """
    # Split dataset into training and test sets (70:30 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "=" * 80)
    print("数据预处理")
    print("=" * 80)
    print(f"\n训练集大小: {X_train.shape[0]} samples")
    print(f"测试集大小: {X_test.shape[0]} samples")

    # Feature standardization (mean=0, std=1)
    scaler: StandardScaler = StandardScaler()
    X_train_scaled: NDArray = scaler.fit_transform(X_train)
    X_test_scaled: NDArray = scaler.transform(X_test)

    print("\n特征标准化完毕.")
    print("训练集 - 平均值:", np.mean(X_train_scaled, axis=0).round(4))
    print("训练集 - 标准差:", np.std(X_train_scaled, axis=0).round(4))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# Task 2: KNN Algorithm Implementation
# ============================================================================


class MyKNN:
    """
    Custom implementation of K-Nearest Neighbors classifier.

    This implementation supports multiple distance metrics and follows
    the standard sklearn-style API with fit() and predict() methods.
    KNN is a lazy learning algorithm that stores training data and
    performs classification based on majority voting of k nearest neighbors.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to consider for classification.
    distance_type : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use for computing distances between samples.

    Attributes
    ----------
    X_train : NDArray of shape (n_samples, n_features)
        Training feature matrix stored during fit().
    y_train : NDArray of shape (n_samples,)
        Training labels stored during fit().

    Examples
    --------
    >>> knn = MyKNN(k=5, distance_type="euclidean")
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, k: int = 5, distance_type: str = "euclidean") -> None:
        """
        Initialize the KNN classifier.

        Parameters
        ----------
        k : int, default=5
            Number of nearest neighbors to consider.
        distance_type : {'euclidean', 'manhattan'}, default='euclidean'
            Distance metric to use.
        """
        self.k: int = k
        self.distance_type: str = distance_type
        self.X_train: NDArray[np.float64] | None = None
        self.y_train: NDArray[np.float64] | None = None

    def fit(
        self, X_train: NDArray[np.float64], y_train: NDArray[np.float64]
    ) -> "MyKNN":
        """
        Fit the KNN model by storing the training data.

        KNN is a lazy learner - no actual training occurs. The method
        simply stores the training data for later use during prediction.

        Parameters
        ----------
        X_train : NDArray[np.float64] of shape (n_samples, n_features)
            Training feature matrix.
        y_train : NDArray[np.float64] of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : MyKNN
            Returns self for method chaining.
        """
        self.X_train = X_train
        self.y_train = y_train
        return self

    def calculate_distance(
        self, x1: NDArray[np.float64], x2: NDArray[np.float64]
    ) -> float:
        """
        Calculate distance between two samples based on the selected metric.

        Parameters
        ----------
        x1 : NDArray of shape (n_features,)
            First sample (feature vector).
        x2 : NDArray of shape (n_features,)
            Second sample (feature vector).

        Returns
        -------
        distance : float
            Computed distance value between x1 and x2.

        Raises
        ------
        ValueError
            If distance_type is not 'euclidean' or 'manhattan'.

        Notes
        -----
        Euclidean distance: d(x1, x2) = sqrt(sum((x1_i - x2_i)^2))
        Manhattan distance: d(x1, x2) = sum(|x1_i - x2_i|)
        """
        if self.distance_type == "euclidean":
            # Euclidean distance: sqrt(sum((x1_i - x2_i)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_type == "manhattan":
            # Manhattan distance: sum(|x1_i - x2_i|)
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def predict_single(self, x: NDArray[np.float64]) -> int:
        """
                Predict the class label for a single test sample.

                Parameters
                ----------
                x : NDArray of shape (n_features,)
                    Single test sample (feature vector).
        NDArray[np.float64]
                Returns
                -------
                label : int
                    Predicted class label.

                Notes
                -----
                The prediction process involves:
                1. Calculate distances from x to all training samples
                2. Sort by distance and select k nearest neighbors
                3. Perform majority voting on neighbor labels
                4. If tie occurs, return label of the closest neighbor
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("You must call fit before predict!")

        # Calculate distances from test sample to all training samples
        distances: list[tuple[float, int]] = []
        for i, train_sample in enumerate(self.X_train):
            dist = self.calculate_distance(x, train_sample)
            distances.append((dist, self.y_train[i]))

        # Sort by distance and select k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest: list[tuple[float, int]] = distances[: self.k]

        # Extract labels of k nearest neighbors
        k_nearest_labels: list[int] = [label for _, label in k_nearest]

        # Majority voting: select the most common label
        label_counts = Counter(k_nearest_labels)
        most_common: list[tuple[int, int]] = label_counts.most_common()

        # Handle tie-breaking: if multiple labels have same count, choose closest
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return k_nearest_labels[0]  # Return label of closest neighbor

        return most_common[0][0]

    def predict(self, X_test: NDArray) -> NDArray:
        """
        Predict class labels for all test samples.

        Parameters
        ----------
        X_test : NDArray of shape (n_test_samples, n_features)
            Test feature matrix.

        Returns
        -------
        predictions : NDArray of shape (n_test_samples,)
            Array of predicted labels for each test sample.
        """
        predictions: list[int] = []
        for x in X_test:
            pred: int = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# Task 3: Model Evaluation and Hyperparameter Tuning
# ============================================================================


def calculate_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for classification using sklearn.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        True labels of the samples.
    y_pred : NDArray of shape (n_samples,)
        Predicted labels from the classifier.
    target_names : list of str
        Names of the classes for reporting purposes.

    Returns
    -------
    metrics : dict
        Dictionary containing the following keys:
        - 'accuracy' : float
            Overall accuracy of predictions.
        - 'confusion_matrix' : np.ndarray of shape (n_classes, n_classes)
            Confusion matrix where rows are true labels and columns are predicted.
        - 'precision' : list of float
            Precision score for each class.
        - 'recall' : list of float
            Recall score for each class.
        - 'f1_scores' : list of float
            F1-score for each class.

    Notes
    -----
    This function uses sklearn.metrics for reliable metric calculations:
    - Accuracy = (TP + TN) / Total
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

    The 'macro' average is not used here; instead, per-class metrics
    are returned as lists to provide detailed performance for each class.
    """
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate confusion matrix
    # rows represent true labels, columns represent predicted labels
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Calculate per-class precision, recall, and F1-score
    # average=None returns scores for each class
    # zero_division=0 handles cases where a class has no predictions
    precision_array = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_array = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_array = f1_score(y_true, y_pred, average=None, zero_division=0)

    if (
        isinstance(precision_array, float | int)
        or isinstance(recall_array, float | int)
        or isinstance(f1_array, float | int)
    ):
        raise ValueError

    # Convert numpy arrays to lists for consistency with original implementation
    precision = precision_array.tolist()
    recall = recall_array.tolist()
    f1_scores = f1_array.tolist()

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_mat,
        "precision": precision,
        "recall": recall,
        "f1_scores": f1_scores,
    }


def print_evaluation_results(metrics: dict[str, Any], target_names: list[str]) -> None:
    """
    Print evaluation metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics from calculate_metrics().
    target_names : list of str
        Names of the classes.

    Returns
    -------
    None
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"\nOverall Accuracy: {metrics['accuracy']:.4f}({metrics['accuracy']*100:.2f}%)"
    )

    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 56)
    for i, name in enumerate(target_names):
        print(
            f"{name:<20} {metrics['precision'][i]:<12.4f} "
            f"{metrics['recall'][i]:<12.4f} {metrics['f1_scores'][i]:<12.4f}"
        )

    print("\nConfusion Matrix:")
    print(f"{'':>15}", end="")
    for name in target_names:
        print(f"{name:<15}", end="")
    print()
    for i, name in enumerate(target_names):
        print(f"{name:>15}", end="")
        for j in range(len(target_names)):
            print(f"{metrics['confusion_matrix'][i, j]:<15}", end="")
        print()


def tune_k_value(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    k_values: list[int],
    distance_type: str = "euclidean",
) -> tuple[int, list[float]]:
    """
    Perform grid search to find optimal k value for KNN classifier.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    k_values : list of int
        List of k values to test during grid search.
    distance_type : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use for KNN.

    Returns
    -------
    best_k : int
        Optimal k value that achieves highest accuracy.
    accuracies : list of float
        List of accuracy values corresponding to each k value tested.

    Notes
    -----
    This function performs exhaustive search over the specified k values,
    evaluating model performance on the test set for each k.
    The k value yielding the highest test accuracy is selected as optimal.
    """
    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER TUNING (Distance: {distance_type})")
    print("=" * 80)

    accuracies: list[float] = []

    for k in k_values:
        knn: MyKNN = MyKNN(k=k, distance_type=distance_type)
        knn.fit(X_train, y_train)
        y_pred: NDArray = knn.predict(X_test)
        accuracy: float = np.mean(y_test == y_pred)
        accuracies.append(accuracy)
        print(f"K={k:2d}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Find best k value
    best_idx = np.argmax(accuracies)
    best_k = k_values[best_idx]
    best_accuracy = accuracies[best_idx]

    print(
        f"\nBest K value: {best_k} with accuracy {best_accuracy:.4f} "
        f"({best_accuracy*100:.2f}%)"
    )

    return best_k, accuracies


# ============================================================================
# Task 4: Visualization
# ============================================================================


def visualize_data_exploration(
    X: NDArray, y: NDArray, feature_names: list[str], target_names: list[str]
) -> None:
    """
    Create comprehensive exploratory data visualizations.

    Parameters
    ----------
    X : NDArray of shape (n_samples, n_features)
        Feature matrix of the dataset.
    y : NDArray of shape (n_samples,)
        Target labels of the dataset.
    feature_names : list of str
        Names of the features.
    target_names : list of str
        Names of the classes.

    Returns
    -------
    None

    Notes
    -----
    This function generates three visualizations:
    1. Feature distribution histograms (4 subplots, one per feature)
    2. Feature correlation heatmap
    3. Scatter plot using two most important features (petal length and width)

    All visualizations are saved as PNG files in the current directory.
    """
    # Create DataFrame for easier plotting
    df: pd.DataFrame = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]

    # Figure 1: Feature distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Distribution by Species", fontsize=16, fontweight="bold")

    colors: list[str] = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for idx, feature in enumerate(feature_names):
        ax = axes[idx // 2, idx % 2]
        for i, species in enumerate(target_names):
            species_data = df[df["species"] == species][feature]
            ax.hist(species_data, alpha=0.6, label=species, bins=15, color=colors[i])
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Distribution of {feature}", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=300, bbox_inches="tight")
    print("\nSaved: feature_distributions.png")

    # Figure 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix: pd.DataFrame = df[feature_names].corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print("Saved: correlation_heatmap.png")

    # Figure 3: Scatter plot (using two most important features)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, species in enumerate(target_names):
        species_data = df[df["species"] == species]
        ax.scatter(
            species_data[feature_names[2]],
            species_data[feature_names[3]],
            label=species,
            alpha=0.7,
            s=100,
            color=colors[i],
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel(feature_names[2], fontsize=12)
    ax.set_ylabel(feature_names[3], fontsize=12)
    ax.set_title(
        "Iris Species Scatter Plot (Petal Features)", fontsize=16, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("scatter_plot.png", dpi=300, bbox_inches="tight")
    print("Saved: scatter_plot.png")


def visualize_model_results(
    metrics: dict[str, Any],
    target_names: list[str],
    k_values: list[int],
    accuracies: list[float],
) -> None:
    """
    Visualize model evaluation results.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics from calculate_metrics().
    target_names : list of str
        Names of the classes.
    k_values : list of int
        List of k values tested during hyperparameter tuning.
    accuracies : list of float
        List of accuracy values corresponding to each k value.

    Returns
    -------
    None

    Notes
    -----
    This function generates two visualizations:
    1. Confusion matrix heatmap
    2. K value vs Accuracy curve with best k marked

    Both visualizations are saved as PNG files in the current directory.
    """
    # Figure 4: Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("\nSaved: confusion_matrix.png")

    # Figure 5: K value vs Accuracy curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        k_values, accuracies, marker="o", linewidth=2, markersize=8, color="#FF6B6B"
    )
    ax.set_xlabel("K Value", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Impact of K Value on Model Accuracy", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)

    # Mark the best k value
    best_idx = np.argmax(accuracies)
    ax.scatter(
        k_values[best_idx],
        accuracies[best_idx],
        color="red",
        s=200,
        zorder=5,
        marker="*",
        label=f"Best K={k_values[best_idx]}",
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("k_value_tuning.png", dpi=300, bbox_inches="tight")
    print("Saved: k_value_tuning.png")


def visualize_decision_boundary(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    knn_model: MyKNN,
    feature_names: list[str],
    target_names: list[str],
    feature_indices: list[int],
) -> None:
    """
    Visualize KNN decision boundaries using two selected features.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    knn_model : MyKNN
        Trained KNN model instance.
    feature_names : list of str
        Names of all features.
    target_names : list of str
        Names of classes.
    feature_indices : list of int, default=[2, 3]
        Indices of two features to use for 2D visualization.
        Default uses petal length (index 2) and petal width (index 3).

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D visualization by:
    1. Extracting two specified features from the dataset
    2. Creating a dense mesh grid covering the feature space
    3. Predicting class for each grid point to visualize decision regions
    4. Overlaying training and test samples on the decision boundary

    The visualization is saved as 'decision_boundary.png'.
    """
    # Extract two features for 2D visualization
    X_train_2d: NDArray = X_train[:, feature_indices]
    X_test_2d: NDArray = X_test[:, feature_indices]

    # Create a temporary KNN model with 2D data
    knn_2d: MyKNN = MyKNN(k=knn_model.k, distance_type=knn_model.distance_type)
    knn_2d.fit(X_train_2d, y_train)

    # Create mesh grid for decision boundary
    x_min: float = X_train_2d[:, 0].min() - 0.5
    x_max: float = X_train_2d[:, 0].max() + 0.5
    y_min: float = X_train_2d[:, 1].min() - 0.5
    y_max: float = X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict for each point in the mesh grid
    grid_points: NDArray = np.c_[xx.ravel(), yy.ravel()]
    Z: NDArray = knn_2d.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(12, 9))
    colors: list[str] = ["#FFE5E5", "#E5F5F5", "#E5EBF5"]

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.4, levels=2, colors=colors)
    ax.contour(
        xx,
        yy,
        Z,
        alpha=0.8,
        levels=2,
        colors=["#FF6B6B", "#4ECDC4", "#45B7D1"],
        linewidths=2,
    )

    # Plot training and test samples
    scatter_colors: list[str] = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for i, species in enumerate(target_names):
        # Training samples
        train_mask: NDArray = y_train == i
        ax.scatter(
            X_train_2d[train_mask, 0],
            X_train_2d[train_mask, 1],
            color=scatter_colors[i],
            label=f"{species} (train)",
            alpha=0.6,
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )

        # Test samples (with different marker)
        test_mask: NDArray = y_test == i
        ax.scatter(
            X_test_2d[test_mask, 0],
            X_test_2d[test_mask, 1],
            color=scatter_colors[i],
            label=f"{species} (test)",
            alpha=0.9,
            s=120,
            marker="^",
            edgecolors="black",
            linewidth=1,
        )

    ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
    ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
    ax.set_title(
        f"KNN Decision Boundary (K={knn_model.k})", fontsize=16, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=300, bbox_inches="tight")
    print("Saved: decision_boundary.png")


# ============================================================================
# Command-Line Interface Functions
# ============================================================================


def run_task1() -> (
    tuple[NDArray, NDArray, NDArray, NDArray, list[str], list[str], StandardScaler]
):
    """
    Execute Task 1: Data loading and preprocessing.

    Returns
    -------
    X_train : NDArray of shape (n_train_samples, n_features)
        Standardized training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Standardized test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    feature_names : list of str
        Names of features.
    target_names : list of str
        Names of classes.
    scaler : StandardScaler
        Fitted scaler object.

    Notes
    -----
    This function performs the complete data pipeline for Task 1:
    - Loads the Iris dataset
    - Explores data statistics
    - Standardizes features
    - Splits into train/test sets
    - Generates exploratory visualizations
    """
    print("\n" + "=" * 80)
    print("EXECUTING TASK 1: DATA LOADING AND PREPROCESSING")
    print("=" * 80)

    # Load and explore data
    X, y, feature_names, target_names = load_and_explore_data()

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Create exploratory visualizations
    visualize_data_exploration(X, y, feature_names, target_names)

    print("\n" + "=" * 80)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return X_train, X_test, y_train, y_test, feature_names, target_names, scaler


def run_task2(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    k: int = 5,
    distance_type: str = "euclidean",
) -> tuple[MyKNN, NDArray]:
    """
    Execute Task 2: KNN algorithm implementation and prediction.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    k : int, default=5
        Number of nearest neighbors.
    distance_type : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use.

    Returns
    -------
    knn_model : MyKNN
        Trained KNN model instance.
    y_pred : NDArray of shape (n_test_samples,)
        Predicted labels for test set.

    Notes
    -----
    This function demonstrates the custom KNN implementation:
    - Initializes MyKNN classifier
    - Fits the model (stores training data)
    - Makes predictions on test set
    - Reports basic accuracy
    """
    print("\n" + "=" * 80)
    print("EXECUTING TASK 2: KNN ALGORITHM IMPLEMENTATION")
    print(f"Parameters: K={k}, Distance={distance_type}")
    print("=" * 80)

    # Initialize and train KNN model
    knn_model: MyKNN = MyKNN(k=k, distance_type=distance_type)
    knn_model.fit(X_train, y_train)

    print(f"\nKNN model trained with K={k} and {distance_type} distance.")
    print("Making predictions on test set...")

    # Make predictions
    y_pred: NDArray = knn_model.predict(X_test)

    # Calculate basic accuracy
    accuracy: float = np.mean(y_test == y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n" + "=" * 80)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return knn_model, y_pred


def run_task3(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    target_names: list[str],
    k_values: list[int],
    distance_type: str = "euclidean",
) -> tuple[int, list[float], dict[str, Any]]:
    """
    Execute Task 3: Model evaluation and hyperparameter tuning.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    target_names : list of str
        Names of classes.
    k_values : list of int, default=[1, 3, 5, 7, 9, 11, 13, 15]
        List of k values to test during tuning.
    distance_type : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use.

    Returns
    -------
    best_k : int
        Optimal k value from grid search.
    accuracies : list of float
        Accuracy values for each tested k.
    metrics : dict
        Comprehensive evaluation metrics for best model.

    Notes
    -----
    This function performs comprehensive model evaluation:
    - Grid search over k values
    - Calculates accuracy, precision, recall, F1-score
    - Generates confusion matrix
    - Identifies optimal k value
    """
    print("\n" + "=" * 80)
    print("EXECUTING TASK 3: MODEL EVALUATION AND HYPERPARAMETER TUNING")
    print("=" * 80)

    # Hyperparameter tuning
    best_k, accuracies = tune_k_value(
        X_train, X_test, y_train, y_test, k_values, distance_type
    )

    # Train final model with best k
    print("\n" + "=" * 80)
    print(f"TRAINING FINAL MODEL WITH BEST K={best_k}")
    print("=" * 80)

    final_knn: MyKNN = MyKNN(k=best_k, distance_type=distance_type)
    final_knn.fit(X_train, y_train)
    y_pred: NDArray = final_knn.predict(X_test)

    # Calculate comprehensive metrics
    metrics: dict[str, Any] = calculate_metrics(y_test, y_pred)
    print_evaluation_results(metrics, target_names)

    print("\n" + "=" * 80)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return best_k, accuracies, metrics


def run_task4(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    feature_names: list[str],
    target_names: list[str],
    best_k: int,
    k_values: list[int],
    accuracies: list[float],
    metrics: dict[str, Any],
    distance_type: str = "euclidean",
) -> None:
    """
    Execute Task 4: Result visualization.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    feature_names : list of str
        Names of features.
    target_names : list of str
        Names of classes.
    best_k : int
        Optimal k value from Task 3.
    k_values : list of int
        List of tested k values.
    accuracies : list of float
        Accuracy values for each k.
    metrics : dict
        Evaluation metrics from Task 3.
    distance_type : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric used.

    Returns
    -------
    None

    Notes
    -----
    This function generates comprehensive visualizations:
    - Confusion matrix heatmap
    - K value tuning curve
    - Decision boundary plot

    All plots are saved as PNG files.
    """
    print("\n" + "=" * 80)
    print("EXECUTING TASK 4: RESULT VISUALIZATION")
    print("=" * 80)

    # Visualize model results
    visualize_model_results(metrics, target_names, k_values, accuracies)

    # Create final model for decision boundary
    final_knn: MyKNN = MyKNN(k=best_k, distance_type=distance_type)
    final_knn.fit(X_train, y_train)

    # Visualize decision boundary
    print("\nGenerating decision boundary visualization...")
    visualize_decision_boundary(
        X_train, X_test, y_train, y_test, final_knn, feature_names, target_names, [2, 3]
    )

    print("\n" + "=" * 80)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)


def run_bonus(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    k_values: list[int],
) -> None:
    """
    Execute bonus task: Compare different distance metrics.

    Parameters
    ----------
    X_train : NDArray of shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : NDArray of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : NDArray of shape (n_train_samples,)
        Training labels.
    y_test : NDArray of shape (n_test_samples,)
        Test labels.
    k_values : list of int, default=[1, 3, 5, 7, 9, 11, 13, 15]
        List of k values to test.

    Returns
    -------
    None

    Notes
    -----
    This bonus task compares Euclidean vs Manhattan distance metrics
    by performing grid search for both and reporting comparative results.
    """
    print("\n" + "=" * 80)
    print("EXECUTING BONUS TASK: DISTANCE METRIC COMPARISON")
    print("=" * 80)

    # Test Manhattan distance
    print("\nTesting Manhattan distance metric...")
    best_k_manhattan, accuracies_manhattan = tune_k_value(
        X_train, X_test, y_train, y_test, k_values, distance_type="manhattan"
    )

    # Test Euclidean distance (for comparison)
    print("\nTesting Euclidean distance metric...")
    best_k_euclidean, accuracies_euclidean = tune_k_value(
        X_train, X_test, y_train, y_test, k_values, distance_type="euclidean"
    )

    # Comparison summary
    print("\n" + "=" * 80)
    print("DISTANCE METRIC COMPARISON SUMMARY")
    print("=" * 80)
    print("\nEuclidean Distance:")
    print(f"  Best K: {best_k_euclidean}")
    print(
        f"  Best Accuracy: {max(accuracies_euclidean):.4f} "
        f"({max(accuracies_euclidean)*100:.2f}%)"
    )

    print("\nManhattan Distance:")
    print(f"  Best K: {best_k_manhattan}")
    print(
        f"  Best Accuracy: {max(accuracies_manhattan):.4f} "
        f"({max(accuracies_manhattan)*100:.2f}%)"
    )

    # Determine which is better
    if max(accuracies_euclidean) > max(accuracies_manhattan):
        print("\nConclusion: Euclidean distance performs better on this dataset.")
    elif max(accuracies_manhattan) > max(accuracies_euclidean):
        print("\nConclusion: Manhattan distance performs better on this dataset.")
    else:
        print("\nConclusion: Both distance metrics perform equally well.")

    print("\n" + "=" * 80)
    print("BONUS TASK COMPLETED SUCCESSFULLY")
    print("=" * 80)


def run_all_tasks() -> None:
    """
    Execute all tasks sequentially (Task 1-4 + Bonus).

    Returns
    -------
    None

    Notes
    -----
    This is the main pipeline that runs the complete project:
    1. Data loading and preprocessing
    2. KNN implementation
    3. Model evaluation and tuning
    4. Result visualization
    5. Bonus: Distance metric comparison
    """
    print("\n" + "=" * 80)
    print("IRIS FLOWER CLASSIFICATION USING K-NEAREST NEIGHBORS")
    print("COMPLETE PROJECT EXECUTION")
    print("=" * 80)

    # Task 1: Data loading and preprocessing
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = run_task1()

    # Task 2: KNN implementation (demo with K=5)
    knn_model, y_pred = run_task2(X_train, X_test, y_train, y_test, k=5)

    # Task 3: Model evaluation and hyperparameter tuning
    k_values: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
    best_k, accuracies, metrics = run_task3(
        X_train, X_test, y_train, y_test, target_names, k_values
    )

    # Task 4: Result visualization
    run_task4(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        target_names,
        best_k,
        k_values,
        accuracies,
        metrics,
    )

    # Bonus: Distance metric comparison
    run_bonus(X_train, X_test, y_train, y_test, k_values)

    # Final summary
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - feature_distributions.png")
    print("  - correlation_heatmap.png")
    print("  - scatter_plot.png")
    print("  - confusion_matrix.png")
    print("  - k_value_tuning.png")
    print("  - decision_boundary.png")
    print("\n" + "=" * 80)


# ============================================================================
# Command-Line Argument Parser
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for task selection.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.

    Notes
    -----
    Supports the following modes:
    - --task1: Run only Task 1 (data loading and preprocessing)
    - --task2: Run only Task 2 (KNN implementation)
    - --task3: Run only Task 3 (evaluation and tuning)
    - --task4: Run only Task 4 (visualization)
    - --bonus: Run only bonus task (distance comparison)
    - --all or no args: Run complete pipeline
    """
    parser = argparse.ArgumentParser(
        description="Iris Flower KNN Classification Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python iris_knn.py                # Run all tasks
  python iris_knn.py --task1        # Run only Task 1
  python iris_knn.py --task2        # Run only Task 2
  python iris_knn.py --task3        # Run only Task 3
  python iris_knn.py --task4        # Run only Task 4
  python iris_knn.py --bonus        # Run only bonus task
  python iris_knn.py --all          # Run all tasks (explicit)
        """,
    )

    parser.add_argument(
        "--task1",
        action="store_true",
        help="Run Task 1: Data loading and preprocessing",
    )

    parser.add_argument(
        "--task2", action="store_true", help="Run Task 2: KNN algorithm implementation"
    )

    parser.add_argument(
        "--task3",
        action="store_true",
        help="Run Task 3: Model evaluation and hyperparameter tuning",
    )

    parser.add_argument(
        "--task4", action="store_true", help="Run Task 4: Result visualization"
    )

    parser.add_argument(
        "--bonus", action="store_true", help="Run Bonus: Distance metric comparison"
    )

    parser.add_argument("--all", action="store_true", help="Run all tasks sequentially")

    parser.add_argument(
        "--k", type=int, default=5, help="K value for Task 2 (default: 5)"
    )

    parser.add_argument(
        "--distance",
        type=str,
        choices=["euclidean", "manhattan"],
        default="euclidean",
        help="Distance metric for Task 2 (default: euclidean)",
    )

    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """
    Main entry point for the Iris KNN classification project.

    Returns
    -------
    None

    Notes
    -----
    Parses command-line arguments and executes the requested tasks.
    If no specific task is selected, runs all tasks by default.
    """
    args: argparse.Namespace = parse_arguments()

    # If no specific task selected, run all tasks
    if (
        not any([args.task1, args.task2, args.task3, args.task4, args.bonus])
        or args.all
    ):
        run_all_tasks()
        return

    # Otherwise, run selected tasks
    # We need to maintain state between tasks, so we'll use a shared data store
    shared_data: dict[str, Any] = {}

    # Task 1
    if args.task1:
        X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
            run_task1()
        )
        shared_data["X_train"] = X_train
        shared_data["X_test"] = X_test
        shared_data["y_train"] = y_train
        shared_data["y_test"] = y_test
        shared_data["feature_names"] = feature_names
        shared_data["target_names"] = target_names
        shared_data["scaler"] = scaler

    # Task 2 (requires Task 1 data)
    if args.task2:
        if "X_train" not in shared_data:
            print("\nWarning: Task 2 requires Task 1 data. Running Task 1 first...")
            X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
                run_task1()
            )
            shared_data["X_train"] = X_train
            shared_data["X_test"] = X_test
            shared_data["y_train"] = y_train
            shared_data["y_test"] = y_test
            shared_data["feature_names"] = feature_names
            shared_data["target_names"] = target_names
            shared_data["scaler"] = scaler

        knn_model, y_pred = run_task2(
            shared_data["X_train"],
            shared_data["X_test"],
            shared_data["y_train"],
            shared_data["y_test"],
            k=args.k,
            distance_type=args.distance,
        )
        shared_data["knn_model"] = knn_model
        shared_data["y_pred"] = y_pred

    # Task 3 (requires Task 1 data)
    if args.task3:
        if "X_train" not in shared_data:
            print("\nWarning: Task 3 requires Task 1 data. Running Task 1 first...")
            X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
                run_task1()
            )
            shared_data["X_train"] = X_train
            shared_data["X_test"] = X_test
            shared_data["y_train"] = y_train
            shared_data["y_test"] = y_test
            shared_data["feature_names"] = feature_names
            shared_data["target_names"] = target_names
            shared_data["scaler"] = scaler

        k_values: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
        best_k, accuracies, metrics = run_task3(
            shared_data["X_train"],
            shared_data["X_test"],
            shared_data["y_train"],
            shared_data["y_test"],
            shared_data["target_names"],
            k_values,
            distance_type=args.distance,
        )
        shared_data["best_k"] = best_k
        shared_data["accuracies"] = accuracies
        shared_data["metrics"] = metrics
        shared_data["k_values"] = k_values

    # Task 4 (requires Tasks 1 and 3 data)
    if args.task4:
        if "X_train" not in shared_data:
            print("\nWarning: Task 4 requires Task 1 data. Running Task 1 first...")
            X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
                run_task1()
            )
            shared_data["X_train"] = X_train
            shared_data["X_test"] = X_test
            shared_data["y_train"] = y_train
            shared_data["y_test"] = y_test
            shared_data["feature_names"] = feature_names
            shared_data["target_names"] = target_names
            shared_data["scaler"] = scaler

        if "best_k" not in shared_data:
            print("\nWarning: Task 4 requires Task 3 results. Running Task 3 first...")
            k_values: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
            best_k, accuracies, metrics = run_task3(
                shared_data["X_train"],
                shared_data["X_test"],
                shared_data["y_train"],
                shared_data["y_test"],
                shared_data["target_names"],
                k_values,
                distance_type=args.distance,
            )
            shared_data["best_k"] = best_k
            shared_data["accuracies"] = accuracies
            shared_data["metrics"] = metrics
            shared_data["k_values"] = k_values

        run_task4(
            shared_data["X_train"],
            shared_data["X_test"],
            shared_data["y_train"],
            shared_data["y_test"],
            shared_data["feature_names"],
            shared_data["target_names"],
            shared_data["best_k"],
            shared_data["k_values"],
            shared_data["accuracies"],
            shared_data["metrics"],
            distance_type=args.distance,
        )

    # Bonus task (requires Task 1 data)
    if args.bonus:
        if "X_train" not in shared_data:
            print("\nWarning: Bonus task requires Task 1 data. Running Task 1 first...")
            X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
                run_task1()
            )
            shared_data["X_train"] = X_train
            shared_data["X_test"] = X_test
            shared_data["y_train"] = y_train
            shared_data["y_test"] = y_test
            shared_data["feature_names"] = feature_names
            shared_data["target_names"] = target_names
            shared_data["scaler"] = scaler

        k_values: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
        run_bonus(
            shared_data["X_train"],
            shared_data["X_test"],
            shared_data["y_train"],
            shared_data["y_test"],
            k_values,
        )


if __name__ == "__main__":
    main()

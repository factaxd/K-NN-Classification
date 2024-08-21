import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.knn import KNearestNeighbors
from src.preprocess import load_and_process_data
from src.evaluation import evaluate_model
import numpy as np
import seaborn as sns

# Adventure theme colors
theme_colors = {
    "background": "#040404",
    "foreground": "#feffff",
    "button": "#417ab3",
    "text": "#eebb6e",
    "cursorColor": "#feffff"
}

# Main window
class KNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-NN Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg=theme_colors['background'])
        
        # Title label
        title_label = tk.Label(root, text="K-NN Classification", font=("Arial", 20), bg=theme_colors['background'], fg=theme_colors['text'])
        title_label.pack(pady=20)

        # Load Dataset button
        load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset, bg=theme_colors['button'], fg=theme_colors['foreground'], font=("Arial", 14))
        load_button.pack(pady=10)

        # Train and Test button
        start_button = tk.Button(root, text="Train and Test Model", command=self.train_and_test_model, bg=theme_colors['button'], fg=theme_colors['foreground'], font=("Arial", 14))
        start_button.pack(pady=10)

        # Result display label
        self.result_label = tk.Label(root, text="", font=("Arial", 14), bg=theme_colors['background'], fg=theme_colors['text'])
        self.result_label.pack(pady=20)

        # Canvas for graph display
        self.canvas = None

    def load_dataset(self):
        # Open file dialog to load dataset
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset_path = file_path
            messagebox.showinfo("Dataset Loaded", f"Dataset successfully loaded from: {file_path}")
        else:
            messagebox.showwarning("Error", "No file selected.")

    def plot_data_distribution(self, X_train, y_train):
        # Plot scatter plot if there are at least 2 features
        if X_train.shape[1] >= 2:
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=40)
            plt.title('Data Distribution')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        else:
            # Plot histogram if there is only one feature
            plt.hist(X_train, bins=20, edgecolor='k')
            plt.title('Feature Distribution')
            plt.xlabel('Feature')
            plt.ylabel('Frequency')
        
        plt.show()

    def plot_confusion_matrix(self, cm, labels):
        """
        Plot the confusion matrix using seaborn's heatmap.
        """
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        return fig, ax

    def train_and_test_model(self):
        try:
            # Load and process data
            X_train, X_test, y_train, y_test = load_and_process_data(self.dataset_path)
            
            # Print data shapes for debugging
            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

            # Plot data distribution
            self.plot_data_distribution(X_train, y_train)

            # Train the KNN model and make predictions
            knn = KNearestNeighbors(k=3)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            
            # Evaluate model performance
            accuracy, cm, report = evaluate_model(y_test, predictions)
            
            # Display results (accuracy and classification report)
            self.result_label.config(text=f"Accuracy: {float(accuracy) * 100:.2f}%\n{report}")

            # Confusion matrix visualization
            if self.canvas:
                self.canvas.get_tk_widget().destroy()  # Remove previous plot
            
            cm_int = cm.astype(int)
            fig, ax = self.plot_confusion_matrix(cm_int, labels=[int(label) for label in np.unique(y_test)])

            # Display confusion matrix on the Tkinter canvas
            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

        except Exception as e:
            import traceback
            traceback.print_exc()  # Detailed traceback for debugging
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()

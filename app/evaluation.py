import numpy as np
import torch
from torch.nn import functional as F
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score, accuracy_score
from result_vis import *
from model import EOGClassifier

class Evaluation:
    def __init__(self, checkpoint_path: str, num_class: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(checkpoint_path, weights_only=False, map_location= self.device)
        self.model = EOGClassifier(num_class = num_class)
        # Load the data to these attributes
        self.data = None
        self.true_labels = None
        self.predicted_labels = None
        self.probabilities = None
        self.prediction_output = None

    def load_weight(self):
        return self.model.load_state_dict(self.checkpoint['model_state_dict']).to(self.device)
    
    def test(self, test_loader: torch.utils.data.DataLoader):
        model = self.load_weight()
        model.eval()
        # Create empty lists to store the predicted labels
        predicted_labels = []
        probabilities = []
        prediction_output = []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                output = model(data)
            probabilities = F.softmax(output, dim = 1)
            _, prediction = torch.max(output, 1)

            # Append the batch predictions to the list
            predicted_labels.extend(prediction.tolist())
            probabilities.extend(probabilities.tolist())
            prediction_output.extend(output.tolist())
            
        # Convert the lists to numpy arrays and torch.Tensor
        self.prediction_output = torch.Tensor(prediction_output)
        self.data = torch.Tensor(test_loader.dataset.data)
        self.true_labels = np.array(test_loader.dataset.labels)
        self.predicted_labels = np.array(predicted_labels)
        self.probabilities = np.array(probabilities)

        return self.true_labels, self.predicted_labels, self.probabilities, self.prediction_output
    
    def main(self):
        self.test()

        # Print classification report and confusion matrix
        print("Cohen Kappa Score:", cohen_kappa_score(self.true_labels, self.predicted_labels))
        print("Accuracy of the model: ", accuracy_score(self.true_labels, self.predicted_labels)*100)
        print("Classification report of the model: \n",classification_report(self.true_labels, self.predicted_labels))
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8,6), dpi=100)
        # Scale up the size of all text
        sns.set(font_scale = 1.1)
        ax = sns.heatmap(cm, annot=True, fmt='.2f', xticklabels = ['wake', 'sleep'], yticklabels = ['wake', 'sleep'],)

        # set x-axis label and ticks.
        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)

        # set y-axis label and ticks
        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)

        # Plot out the probability and roc curve of each class
        probability_roc_curve(test_tensor= self.data,
                              predicted_labels= self.predicted_labels,
                              probabilities= self.probabilities,
                              num_classes= 5)   

        # Plot out the entropy
        plot_entropy(prediction_output= self.prediction_output)

        # Plot out rhe class distribution
        plot_class_distribution(predicted_labels= self.predicted_labels,
                                num_classes= 5)
        
        # Plot out the confidence histogram
        plot_confidence_hist(prediction_output= self.prediction_output)

        # Plot out the violin plot
        plot_violin(prediction_output= self.prediction_output,
                    predicted_labels= self.predicted_labels,
                    num_classes= 5)

        # Plot out the 3D scatter plot
        plot_3d_scatter(prediction_output= self.prediction_output,
                        predicted_labels= self.predicted_labels,
                        num_classes= 5)
        plt.show()

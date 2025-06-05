import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pandas as pd
import torch
from train_model import DetectionModel
import importlib
# import yolov9.detect_dual as detect_dual
# importlib.reload(detect_dual)
import importlib
import pose_estimation
importlib.reload(pose_estimation)
from pose_estimation import detect_pose
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
# from merge_vedio import detect_double_and_four_hits
# classification_report
from sklearn.metrics import classification_report, confusion_matrix

TEST_DATA_PATH = './csv/test_result.csv'
model_path = '/home/viplab/dodofang/Auto-scoring-System-for-Volleyball-Matches-/result/refereeSignalsModel_20250602223026.pth'
# actions = ['01_left', '02_right', '03_outball', '04_ball_in', '05_timeout', '06_ball_touch', '07_penetration_under_the_net', '08_catch', 
#             '09_end_of_setmatch', '10_substitusion', '11_change_courts', '12_four_hits', '13_net_touch', '14_serve', '15_double_contact', '16_inbound']
actions = ['01_left', '02_right', '03_outball', '04_ball_in', '05_timeout', '07_penetration_under_the_net', '08_catch', 
                  '09_end_of_setmatch', '10_substitusion', '11_change_courts', '12_four_hits', '13_net_touch', '14_serve', '15_double_contact', '16_inbound', '17_ball_touch_2']
num_actions = len(actions)
batch_size = 8
timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

def load_data(file_path):
    """
    Load data from the specified file.
    Assumes the file is in a CSV-like format, where each line starts with a label
    and then a series of floating point numbers representing features.
    """
    data = []
    labels = []
    filenames = []
    with open(file_path, 'r') as f:
        for line in f:
            # skip first line
            if line.startswith('0,') or line.startswith('0\t'):
                continue
            parts = line.strip().split(',')
            label = parts[0]  # The first element is the label
            filenames.append(parts[1])
            features = parts[35:77] + parts[101:]  # The rest are feature values
            # print(features)
            data.append([float(f) for f in features])

            # Convert label to a one-hot encoding or an integer label
            # Adjust the label conversion as per your specific use case
            label_idx = actions.index(label) if label in actions else -1
            if label_idx != -1:
                labels.append(label_idx)
    return data, filenames, labels

def draw_confusion_matrix(title, all_labels, all_preds):
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_actions))
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    # 绘制混淆矩阵的热图
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Blues", xticklabels=actions, yticklabels=actions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
    plt.savefig(f'./result/{title}_confusion_matrix_{timestamp}.png')

def main():
    # Load and preprocess test data
    print('Loading test data...')
    sequences, filenames, labels = load_data(TEST_DATA_PATH)
    X = torch.tensor(np.array(sequences, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Load the trained model
    model = DetectionModel(num_actions=len(actions))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Testing loop
    from sklearn.metrics import precision_score, recall_score, f1_score

    testing_results = []

    all_labels = []
    all_preds = []
    loss = 0.0
    correct_preds = 0
    total_preds = 0
    criterion = torch.nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for classification

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # Compute loss
            loss += criterion(outputs, labels).item() * inputs.size(0)

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            if predicted == '15_double_contact' or predicted == '12_four_hits':
                predicted = detect_double_and_four_hits(result[0][101:(101 + 63 + 63)])
            
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    loss = loss / len(test_loader.dataset)
    accuracy = correct_preds / total_preds
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    testing_results.append((loss, accuracy, precision, recall, f1))
    
    # print the result
    print(f'(Testing), Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, '
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    with open(f'./result/log_{timestamp}.txt', 'a') as f:
        f.write(f'(Testing), Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, '
                f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n')

    draw_confusion_matrix('testing', all_labels, all_preds)
    
    # Generate classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=actions, zero_division=0))

    # Generate confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    

    # Save misclassified examples
    misclassified_file = './result/misclassified_test.txt'
    with open(misclassified_file, 'w') as f:
        for i in range(len(all_labels)):
            if all_labels[i] != all_preds[i]:
                f.write(f'Filename: {filenames[i]}, True label: {actions[all_labels[i]]}, Predicted: {actions[all_preds[i]]}\n')
    print(f"Misclassified examples saved to {misclassified_file}")
    
if __name__ == '__main__':
    main()

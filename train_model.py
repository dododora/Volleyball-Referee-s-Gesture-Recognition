import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

actions_labels = ['01_left', '02_right', '03_outball', '04_ball_in', '05_timeout', '07_penetration_under_the_net', '08_catch', 
                  '09_end_of_setmatch', '10_substitusion', '11_change_courts', '12_four_hits', '13_net_touch', '14_serve', '15_double_contact', '16_inbound', '17_ball_touch_2']

DATA_PATH = './csv/'
TRAIN_DATA_PATH = DATA_PATH + 'train_result.csv'
TEST_DATA_PATH = DATA_PATH + 'test_result.csv'
VALID_DATA_PATH = DATA_PATH + 'valid_result.csv'

actions = ['01_left', '02_right', '03_outball', '04_ball_in', '05_timeout', '07_penetration_under_the_net', '08_catch', 
                  '09_end_of_setmatch', '10_substitusion', '11_change_courts', '12_four_hits', '13_net_touch', '14_serve', '15_double_contact', '16_inbound', '17_ball_touch_2']
num_actions = len(actions)
num_epochs = 300
batch_size = 8
timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

class DetectionModel(torch.nn.Module):
    def __init__(self, num_actions, input_size=198):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.norm1 = torch.nn.LayerNorm([128, 3])  
        self.dropout = torch.nn.Dropout(p=0.3)
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(128 * 2, num_actions)



    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() >= 3:
            x = x.transpose(1, 2)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.norm1(x)  # 使用修改過的 LayerNorm
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        # out = torch.softmax(self.fc(x),dim=1)
        out = torch.sigmoid(self.fc(x))
        return out

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
            label_idx = actions_labels.index(label) if label in actions_labels else -1
            if label_idx != -1:
                labels.append(label_idx)
    return data, filenames, labels

def draw_confusion_matrix(title, all_labels, all_preds):
    # 提取所有标签和预测结果
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_actions))
    # print(cm)
    # print(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    # 绘制混淆矩阵的热图
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Blues", xticklabels=actions, yticklabels=actions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
    plt.savefig(f'./result/{title}_confusion_matrix_{timestamp}.png')


    ## 畫圖

def draw_plot(title, training_results):
    losses, accuracies, precisions, recalls, f1_scores = zip(*training_results)

    # 绘制训练指标的曲线
    plt.figure(figsize=(12, 10))
    
    # 绘制损失的变化曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses, label='Loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # 绘制准确率的变化曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # 绘制精确度、召回率和 F1-score 的变化曲线
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), precisions, label='Precision', color='tab:green')
    plt.plot(range(1, num_epochs + 1), recalls, label='Recall', color='tab:orange')
    plt.plot(range(1, num_epochs + 1), f1_scores, label='F1-Score', color='tab:purple')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score')
    plt.legend()

    # save figure
    plt.savefig(f'./result/{title}_{timestamp}.png')

    # # 显示图表
    # plt.tight_layout()
    # plt.show()

def main():
    # Load and preprocess data
    print('Loading data...')
    sequences, filenames, labels = load_data(TRAIN_DATA_PATH)
    val_sequences, val_filenames, val_labels = load_data(VALID_DATA_PATH)
    print(labels)
    # Convert to tensors
    # Delete firstline
    X = torch.tensor(np.array(sequences, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)  # Use long for classification labels
    print(X.shape, y.shape)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_X = torch.tensor(val_sequences, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.long)  # Use long for classification labels
    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Initialize the model
    model = DetectionModel(num_actions=len(actions))

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    training_results = []
    validation_results = []
    training_all_labels = []
    training_all_preds = []
    # Training loop
    from sklearn.metrics import precision_score, recall_score, f1_score
    import os

    for epoch in range(num_epochs):
        # ======== 訓練階段 ========
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_labels = []  # 用来存储所有标签
        all_preds = []   # 用来存储所有预测结果

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Get predictions and calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            # Check if the gesture is double contact and the predicted probability for four hits
            # for i in range(len(predicted)):
            #     if predicted[i] == actions.index('15_double_contact'):
            #         four_hits_prob = outputs[i][actions.index('12_four_hits')].item()
            #         # print('aaaaaaaaaaa', four_hits_prob)
            #         if four_hits_prob > 0.6:
            #             # print('aaaaaaaaaaa', four_hits_prob)
            #             predicted[i] = actions.index('12_four_hits')

            # 将所有的标签和预测结果存储起来
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds / total_preds
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        # 訓練結果保存
        training_results.append((epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1))
        if epoch == num_epochs - 1:
            draw_confusion_matrix('training', all_labels, all_preds)
        with open(f'./result/log_{timestamp}.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{num_epochs}] (Training), Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, '
                    f'Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1-Score: {epoch_f1:.4f}\n')
        print(f'Epoch [{epoch+1}/{num_epochs}] (Training), Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, '
            f'Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1-Score: {epoch_f1:.4f}')
        
        # ======== 驗證階段 ========
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0
        val_all_labels = []  # 用来存储所有验证标签
        val_all_preds = []   # 用来存储所有验证预测结果

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                
                # Compute loss
                val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)

                # Get predictions and calculate accuracy
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct_preds += (val_predicted == val_labels).sum().item()
                val_total_preds += val_labels.size(0)
                # Check if the gesture is double contact and the predicted probability for four hits
                for i in range(len(predicted)):
                    if predicted[i] == actions.index('15_double_contact'):
                        four_hits_prob = outputs[i][actions.index('12_four_hits')].item()
                        # print('aaaaaaaaaaa', four_hits_prob)
                        if four_hits_prob > 0.6:
                            # print('aaaaaaaaaaa', four_hits_prob)
                            predicted[i] = actions.index('12_four_hits')

                # 将所有的验证标签和预测结果存储起来
                val_all_labels.extend(val_labels.cpu().numpy())
                val_all_preds.extend(val_predicted.cpu().numpy())
                # Print misclassified examples
                misclassified_file = './result/misclassified.txt'
                # print( val_inputs)
                if epoch == num_epochs - 1:
                    for i in range(len(val_labels)):
                        if val_labels[i] != val_predicted[i]:
                            # Get the filename of the misclassified example
                            # filename = val_filenames[val_total_preds - len(val_labels) + i]
                            filename = val_filenames[i]
                            # print(f'[{filename}] True label: {actions[val_labels[i]]}, Predicted: {actions[val_predicted[i]]}')
                            with open(misclassified_file, 'a') as f:
                                # Write the filename to the file
                                f.write(f'[{filename}] True label: {actions[val_labels[i]]}, Predicted: {actions[val_predicted[i]]}\n')

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_accuracy = val_correct_preds / val_total_preds
        val_epoch_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        val_epoch_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        val_epoch_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        validation_results.append((val_epoch_loss, val_epoch_accuracy, val_epoch_precision, val_epoch_recall, val_epoch_f1))
        # 验证结果打印
        print(f'Epoch [{epoch+1}/{num_epochs}] (Validation), Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}, '
            f'Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1-Score: {val_epoch_f1:.4f}')
        with open(f'./result/log_{timestamp}.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{num_epochs}] (Validation), Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}, '
                    f'Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1-Score: {val_epoch_f1:.4f}\n')

        if epoch == num_epochs - 1:
            draw_confusion_matrix('validation', val_all_labels, val_all_preds)

    # Save the trained model
    torch.save(model.state_dict(), f'./result/refereeSignalsModel_{timestamp}.pth')
    draw_plot('training_results', training_results)
    draw_plot('validation_results', validation_results)

if __name__ == '__main__':
    main()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-07T16:11:36.255588Z","iopub.execute_input":"2026-01-07T16:11:36.255907Z","iopub.status.idle":"2026-01-07T16:11:39.538086Z","shell.execute_reply.started":"2026-01-07T16:11:36.255880Z","shell.execute_reply":"2026-01-07T16:11:39.537192Z"}}
# Install the grad-cam library
!pip install grad-cam

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:36:07.777224Z","iopub.execute_input":"2026-01-07T17:36:07.777950Z","iopub.status.idle":"2026-01-07T17:36:07.786185Z","shell.execute_reply.started":"2026-01-07T17:36:07.777908Z","shell.execute_reply":"2026-01-07T17:36:07.785587Z"}}
# Import necessary libraries
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import seaborn as sns
import os
import copy
from torch.autograd import Variable
from collections import Counter
import random

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:36:08.308435Z","iopub.execute_input":"2026-01-07T17:36:08.308911Z","iopub.status.idle":"2026-01-07T17:36:08.314256Z","shell.execute_reply.started":"2026-01-07T17:36:08.308870Z","shell.execute_reply":"2026-01-07T17:36:08.313692Z"}}

# Define classes to use and create class mappings
selected_classes = [3, 5, 7]
class_map = {3: 0, 5: 1, 7: 2}
num_classes = len(selected_classes)
class_names = ['cat', 'dog', 'horse']

# Define training data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define validation/test data normalization (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:36:08.586028Z","iopub.execute_input":"2026-01-07T17:36:08.586238Z","iopub.status.idle":"2026-01-07T17:37:27.830063Z","shell.execute_reply.started":"2026-01-07T17:36:08.586218Z","shell.execute_reply":"2026-01-07T17:37:27.829296Z"}}

# Load full CIFAR-10 dataset
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

# Get labels from datasets to check class distribution
train_labels = [label for _, label in full_trainset]
test_labels = [label for _, label in full_testset]

print("Train class counts:", Counter(train_labels))
print("Test class counts:", Counter(test_labels))

# Display a sample image from the dataset
sample_img, sample_label = full_trainset[0]
img_display = sample_img.permute(1, 2, 0).numpy()
img_display = (img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

plt.imshow(img_display)
plt.title(f"Sample: Class {full_trainset.classes[sample_label]}")
plt.axis('off')
plt.show()


# %% [markdown]
# **All CIFAR-10 classes balanced** - 5,000 train and 1,000 test samples per class

# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:37:27.831670Z","iopub.execute_input":"2026-01-07T17:37:27.832049Z","iopub.status.idle":"2026-01-07T17:38:58.768725Z","shell.execute_reply.started":"2026-01-07T17:37:27.832019Z","shell.execute_reply":"2026-01-07T17:38:58.767701Z"}}

# Create custom dataset class to filter only selected classes
class FilteredCIFAR10(Dataset):
    def __init__(self, dataset, selected_classes, class_map):
        self.dataset = dataset
        self.indices = []

        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in selected_classes:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, class_map[label]


# Apply filtering to train and test datasets
trainset_filtered = FilteredCIFAR10(full_trainset, selected_classes, class_map)
testset_filtered = FilteredCIFAR10(full_testset, selected_classes, class_map)



# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:38:58.770043Z","iopub.execute_input":"2026-01-07T17:38:58.770414Z","iopub.status.idle":"2026-01-07T17:39:41.268096Z","shell.execute_reply.started":"2026-01-07T17:38:58.770362Z","shell.execute_reply":"2026-01-07T17:39:41.267274Z"}}

# Set maximum number of samples and split ratios
max_total = 5000
train_ratio = 0.8
per_class_total = max_total // num_classes
per_class_train = int(per_class_total * train_ratio)
per_class_test = per_class_total - per_class_train

# Function to get indices for each class
def get_class_indices(dataset):
    class_indices = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_indices[label].append(i)
    return class_indices

# Get class indices for balanced sampling
train_class_indices = get_class_indices(trainset_filtered)
test_class_indices = get_class_indices(testset_filtered)

# Sample equal number of examples per class
train_indices = []
test_indices = []

for c in range(num_classes):
    train_indices += random.sample(train_class_indices[c], per_class_train)
    test_indices += random.sample(test_class_indices[c], per_class_test)

# Create subsets with balanced classes
trainset_filtered = Subset(trainset_filtered, train_indices)
testset_filtered = Subset(testset_filtered, test_indices)

print(f"Train: {len(trainset_filtered)}, Test: {len(testset_filtered)}, Total: {len(trainset_filtered)+len(testset_filtered)}")


# Verify class distribution in filtered datasets
filtered_train_labels = [trainset_filtered[i][1] for i in range(len(trainset_filtered))]
filtered_test_labels = [testset_filtered[i][1] for i in range(len(testset_filtered))]

print("\nFiltered train class counts:", Counter(filtered_train_labels))
print("Filtered test class counts:", Counter(filtered_test_labels))
print(f"Total filtered train samples: {len(trainset_filtered)}")
print(f"Total filtered test samples: {len(testset_filtered)}")

# Display one sample image from each class
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, class_label in enumerate([0, 1, 2]):
    for i in range(len(trainset_filtered)):
        img, label = trainset_filtered[i]
        if label == class_label:
            img_display = img.permute(1, 2, 0).numpy()
            img_display = (img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            axes[idx].imshow(img_display)
            axes[idx].set_title(f"Class {class_label}: {class_names[class_label]}")
            axes[idx].axis('off')
            break

plt.tight_layout()
plt.show()


# %% [markdown]
# **Balanced 3-class subset created** - Total 4,998 samples (≤5,000 requirement met)
# **Equal class distribution** - Each class has ~1,332 train and 334 test samples

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:39:41.269877Z","iopub.execute_input":"2026-01-07T17:39:41.270188Z","iopub.status.idle":"2026-01-07T17:39:47.732202Z","shell.execute_reply.started":"2026-01-07T17:39:41.270161Z","shell.execute_reply":"2026-01-07T17:39:47.731285Z"}}

# Split training data into train and validation sets using stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

labels = [trainset_filtered[i][1] for i in range(len(trainset_filtered))]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

train_dataset = Subset(trainset_filtered, train_idx)
val_dataset = Subset(trainset_filtered, val_idx)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:39:47.733354Z","iopub.execute_input":"2026-01-07T17:39:47.733644Z","iopub.status.idle":"2026-01-07T17:39:47.741490Z","shell.execute_reply.started":"2026-01-07T17:39:47.733615Z","shell.execute_reply":"2026-01-07T17:39:47.740609Z"}}
# Create data loaders for batching
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset_filtered, batch_size=batch_size, shuffle=False)

print(f"\nTraining samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(testset_filtered)}")


# %% [markdown]
# 
# **Stratified 80/20 split** - 3,196 train, 800 validation, 1,002 test samples

# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:47:47.831875Z","iopub.execute_input":"2026-01-07T17:47:47.832477Z","iopub.status.idle":"2026-01-07T17:47:48.040713Z","shell.execute_reply.started":"2026-01-07T17:47:47.832443Z","shell.execute_reply":"2026-01-07T17:47:48.039948Z"}}

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classifier
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for our 3 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:54:56.416422Z","iopub.execute_input":"2026-01-07T17:54:56.417255Z","iopub.status.idle":"2026-01-07T17:57:27.456953Z","shell.execute_reply.started":"2026-01-07T17:54:56.417220Z","shell.execute_reply":"2026-01-07T17:57:27.456262Z"}}

# Define training function with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:57:27.458423Z","iopub.execute_input":"2026-01-07T17:57:27.458704Z","iopub.status.idle":"2026-01-07T17:57:27.550716Z","shell.execute_reply.started":"2026-01-07T17:57:27.458682Z","shell.execute_reply":"2026-01-07T17:57:27.550096Z"}}

# Save the trained model
import os
save_path = '/kaggle/working/model.pth' if 'kaggle' in os.getcwd() else 'model.pth'
torch.save(model.state_dict(), save_path)
print(f"Best model saved to {save_path}")


# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:57:27.551562Z","iopub.execute_input":"2026-01-07T17:57:27.551872Z","iopub.status.idle":"2026-01-07T17:57:30.411849Z","shell.execute_reply.started":"2026-01-07T17:57:27.551839Z","shell.execute_reply":"2026-01-07T17:57:30.411203Z"}}

# Define evaluation function to test model performance
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    correct_examples = []
    incorrect_examples = []
    max_correct = 20
    max_incorrect = 20
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect examples for visualization
            for i in range(inputs.size(0)):
                if preds[i] == labels[i] and len(correct_examples) < max_correct:
                    correct_examples.append((inputs[i].cpu(), preds[i].cpu(), labels[i].cpu()))
                elif preds[i] != labels[i] and len(incorrect_examples) < max_incorrect:
                    incorrect_examples.append((inputs[i].cpu(), preds[i].cpu(), labels[i].cpu()))
    
    # Calculate and display accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nTest Accuracy: {accuracy:.4f}')
    
    # Display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Display correct predictions
    print(f'\n{min(5, len(correct_examples))} Correct Predictions:')
    fig, axes = plt.subplots(1, min(5, len(correct_examples)), figsize=(15, 3))
    if len(correct_examples) == 1:
        axes = [axes]
    
    for i, (img, pred, label) in enumerate(correct_examples[:5]):
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0,1)
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {class_names[pred]}\nTrue: {class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display incorrect predictions
    print(f'{min(5, len(incorrect_examples))} Incorrect Predictions:')
    fig, axes = plt.subplots(1, min(5, len(incorrect_examples)), figsize=(15, 3))
    if len(incorrect_examples) == 1:
        axes = [axes]
    
    for i, (img, pred, label) in enumerate(incorrect_examples[:5]):
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0,1)
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {class_names[pred]}\nTrue: {class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed classification report
    from sklearn.metrics import classification_report
    print("\nPer-Class Performance:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return correct_examples, incorrect_examples

# Evaluate the model on test set
correct_examples, incorrect_examples = evaluate_model(model, test_loader, class_names)


# %% [markdown]
# **83% Test Accuracy achieved**
# **Horse class performs best** (90% F1-score)
# **Cat class slightly weaker** (78% F1-score) - possible confusion with dog features
# **Balanced performance** across all classes (macro avg = 0.83)

# %% [code] {"execution":{"iopub.status.busy":"2026-01-07T17:57:30.413493Z","iopub.execute_input":"2026-01-07T17:57:30.413777Z","iopub.status.idle":"2026-01-07T17:57:32.871291Z","shell.execute_reply.started":"2026-01-07T17:57:30.413755Z","shell.execute_reply":"2026-01-07T17:57:32.870730Z"}}

# Generate Grad-CAM visualizations to understand model decisions
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

model.eval()
model.layer4.requires_grad_(True)

# Function to denormalize images for visualization
def prepare_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = img_tensor.cpu().numpy()
    img = img * std + mean
    img = np.transpose(img, (1,2,0))
    return np.clip(img, 0, 1)

# Target the last layer of ResNet18 for Grad-CAM
target_layers = [model.layer4[-1]]

with GradCAM(model=model, target_layers=target_layers) as cam:
    # Visualize correct predictions
    for i, (img_tensor, pred, true) in enumerate(correct_examples[:5]):
        cam_map = cam(
            input_tensor=img_tensor.unsqueeze(0).to(device),
            targets=[ClassifierOutputTarget(pred)]
        )[0]
        
        img_np = prepare_image(img_tensor)
        overlay = show_cam_on_image(img_np, cam_map, use_rgb=True)
        
        plt.figure(figsize=(10,3))
        
        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title(f"Original\nTrue: {class_names[true]}")
        plt.axis("off")
        
        plt.subplot(1,3,2)
        plt.imshow(cam_map, cmap="jet")
        plt.title("Grad-CAM")
        plt.axis("off")
        
        plt.subplot(1,3,3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPred: {class_names[pred]}")
        plt.axis("off")
        
        plt.suptitle("CORRECT Prediction", fontsize=13)
        plt.tight_layout()
        plt.show()
        
        focus_intensity = np.mean(cam_map)
        print(f"Image {i+1} - Interpretation: Strong focus (intensity: {focus_intensity:.2f}) on key features like the {class_names[true].lower()}'s shape, leading to accurate prediction.")
    
    # Visualize incorrect predictions
    for i, (img_tensor, pred, true) in enumerate(incorrect_examples[:5]):
        cam_map = cam(
            input_tensor=img_tensor.unsqueeze(0).to(device),
            targets=[ClassifierOutputTarget(pred)]
        )[0]
        
        img_np = prepare_image(img_tensor)
        overlay = show_cam_on_image(img_np, cam_map, use_rgb=True)
        
        true_cam_map = cam(
            input_tensor=img_tensor.unsqueeze(0).to(device),
            targets=[ClassifierOutputTarget(true)]
        )[0]
        
        true_overlay = show_cam_on_image(img_np, true_cam_map, use_rgb=True)
        
        plt.figure(figsize=(13,3))
        
        plt.subplot(1,4,1)
        plt.imshow(img_np)
        plt.title(f"Original\nTrue: {class_names[true]}")
        plt.axis("off")
        
        plt.subplot(1,4,2)
        plt.imshow(cam_map, cmap="jet")
        plt.title("Pred Grad-CAM")
        plt.axis("off")
        
        plt.subplot(1,4,3)
        plt.imshow(overlay)
        plt.title(f"Pred Overlay\nPred: {class_names[pred]}")
        plt.axis("off")
        
        plt.subplot(1,4,4)
        plt.imshow(true_overlay)
        plt.title(f"True Overlay\nTrue: {class_names[true]}")
        plt.axis("off")
        
        plt.suptitle("INCORRECT Prediction", fontsize=13)
        plt.tight_layout()
        plt.show()
        
        focus_intensity = np.mean(cam_map)
        print(f"Image {i+1} - Interpretation: Misplaced focus (intensity: {focus_intensity:.2f}) possibly on background or similar features, causing confusion with {class_names[pred].lower()}.")

# %% [markdown]
# Here's a concise markdown version:
# 
# ---
# 
# ## **Grad-CAM Explainability Analysis**
# 
# ### **✅ Correct Predictions - What Works**
# - **Strong focused attention** (intensity ~0.3-0.4) on key features: animal shape, head, body outline
# - **Foreground isolation** - attention stays on subject, ignores background
# - **Class-specific traits captured** - elongated bodies (horse), compact forms (cat/dog)
# 
# ---
# 
# ### **❌ Incorrect Predictions - Where Model Fails**
# 
# **1. Localized Focus Problem**
# - Attention on small regions (body mass only), misses limbs/tails/face
# - Fails to capture holistic shape → confuses similar classes (cat ↔ dog)
# 
# **2. Background Interference**
# - Attention spills to irrelevant areas (environment, objects)
# - Learns spurious correlations, not true animal features
# 
# **3. Texture Over Structure**
# - Prioritizes superficial cues (fur patterns, color)
# - Ignores structural differences (ear shape, posture)
# - Common cat/dog confusion due to similar textures
# 
# **4. Poor Image Quality**
# - Blurry/low-res images → diffuse, weak attention
# - Can't lock onto meaningful features → higher error rate
# 
# ---

# %% [code]

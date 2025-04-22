import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from classifier import ImageClassifier
from pipeline import create_data_loaders

def evaluate_model(model_path, data_dir, num_classes, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = ImageClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Create data loader for validation set
    _, val_loader, class_to_idx = create_data_loaders(
        data_dir, batch_size=batch_size
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('evaluation_confusion_matrix.png')
    
    # Print classification report
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names))
    
    return {
        'confusion_matrix': cm,
        'classification_report': classification_report(
            all_labels, all_preds, 
            target_names=class_names, 
            output_dict=True
        )
    }
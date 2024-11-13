import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression

obj_classes = {
    "002_master_chef_can": 0,
    "003_cracker_box": 1,
    "004_sugar_box": 2,
    "005_tomato_soup_can": 3,
    "006_mustard_bottle": 4,
    "007_tuna_fish_can": 5,
    "008_pudding_box": 6,
    "009_gelatin_box": 7,
    "010_potted_meat_can": 8,
    "011_banana": 9,
    "019_pitcher_base": 10,
    "021_bleach_cleanser": 11,
    "024_bowl": 12,
    "025_mug": 13,
    "035_power_drill": 14,
    "036_wood_block": 15,
    "037_scissors": 16,
    "040_large_marker": 17,
    "051_large_clamp": 18,
    "052_extra_large_clamp": 19,
    "061_foam_brick": 20
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class YCBDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="", transform=None, is_train=True):
        self.data_dir = data_dir
        self.data_idx = np.loadtxt(f"{self.data_dir}/data.txt", delimiter=",", dtype=int)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return self.data_idx.shape[0]

    def _load_labels(self, fname):
        '''
        Reads label
        '''
        f = open(fname, 'r')
        data = f.readlines()
        labels = []
        for line  in data:
            l = line.split(' ')[0]
            labels.append(l)
        f.close()
        return labels
        
    def __getitem__(self, index):
        v_num, t_num = self.data_idx[index]
        
        img_id = f"{v_num:04}_{t_num:06}"
        fname = f"{self.data_dir}/{v_num:04}/{t_num:06}"

        X = torchvision.io.read_image(f"{fname}-color.png")

        # preprocess for size
        preprocess = transforms.ConvertImageDtype(torch.float32)
        X_original = preprocess(X)

        if self.transform is not None:
            X_transformed = self.transform(X_original)
        else:
            X_transformed = X_original

        if not self.is_train: # no labels are provided for testing set
            y = torch.FloatTensor(np.ones(len(obj_classes))*(-1.0))
        else:
            # load string labels and convert to k-hot labels
            string_labels = self._load_labels(f"{fname}-box.txt")
            y = np.zeros(len(obj_classes))
            for l in string_labels:
                y[obj_classes[l]] = 1.0
            y = torch.FloatTensor(np.array(y))

        return img_id, X_original, X_transformed, y

train_dataset = YCBDataset(data_dir="/kaggle/input/cs3264-assignment-2-ay2425s1/ycb_dataset/train_data", transform=transform, is_train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)

test_dataset = YCBDataset(data_dir="/kaggle/input/cs3264-assignment-2-ay2425s1/ycb_dataset/test_data", transform=transform, is_train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)

print(train_dataset.__len__())

img_id, X_original, X_transformed, y = train_dataset[0]
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(F.to_pil_image(X_original))
ax[1].imshow(F.to_pil_image(X_transformed))
ax[0].title.set_text('Original Image')
ax[1].title.set_text('Transformed Image')
plt.show()

print(y)
print(X_transformed.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add your Neural Network here (if using NN)
net = resnet18(weights=ResNet18_Weights.DEFAULT)
net.fc = nn.Identity() # Replace last layer to keep the 2nd last layer output

net.to(device)

criterion = None
optimizer = None

# To find out all features + corresponding labels
resnet_output_arr = []

class_arr = []
for i in range(21):
    class_arr.append([])

with torch.no_grad():
    for img_id, X_original, X_transformed, y in train_dataloader:
        X_transformed = X_transformed.to(device)
        resnet_outputs = net(X_transformed)
        
        for output in resnet_outputs:
            resnet_output_arr.append(output.cpu())

        for y_indiv in y:
            for i in range(21):
                class_arr[i].append(y_indiv[i])
    
resnet_output_arr = np.array(resnet_output_arr)
# for i in range(21):
#     class_arr[i] = np.array(class_arr[i])
class_arr = np.array(class_arr)

logistic_reg_models = []
    
print(type(class_arr))
print(class_arr.shape)
print(class_arr[0])
print(class_arr[0].shape)

print(type(resnet_output_arr))
print(type(resnet_output_arr[0]))
print(resnet_output_arr.shape)

logistic_regression_models = []

for i in range(21):
    lr_model = LogisticRegression(solver="liblinear")
    lr_model.fit(resnet_output_arr, class_arr[i])
    logistic_regression_models.append(lr_model)

def to_csv(fname, img_ids=None, results=None):
    '''
    Utility function to save submission csv
    input:
        - img_ids, list of img_ids.
        - results, list of predictions for multi-label binarization.
    '''
    results = np.array(results).astype(int)
    df = pd.DataFrame([pd.Series(x) for x in results])
    df.columns = ['class_{}'.format(x) for x in df.columns]
    df = df.assign(img_id = img_ids)

    cols = df.columns.to_list()
    df = df[[cols[-1]]+cols[:-1]]

    df.to_csv(fname, index=False)

with torch.no_grad():
    img_ids = []
    results = []
    for img_id, X_original, X_transformed, _ in test_dataloader:
        X_transformed = X_transformed.to(device)
        resnet_outputs = net(X_transformed).cpu()

        pred_matrix = []
        
        for i in range(21):
            lr_model = logistic_regression_models[i]
            pred = lr_model.predict(resnet_outputs)
            pred_matrix.append(pred)

        pred_matrix = np.array(pred_matrix).T

        for y_pred in pred_matrix:
            results.append(y_pred)
            
        for i in range(X_transformed.shape[0]):
            img_ids.append(img_id[i])

print(results[0])
print(np.array(results).shape)
print(img_ids)
print(np.array(img_ids).shape)

to_csv("submission.csv", img_ids=img_ids, results=results)

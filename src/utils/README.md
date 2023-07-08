### Example of using the custom AudioEmotionDataset class.


```
import torch.nn as nn
import os
import pandas as pd
from src.utils.dataset import AudioEmotionDataset
from src.utils.constants import Columns, Emotions, Datasets
import src.utils.data_processing as data_processing

ROOT = 'your/path/to/project/root/with/data/folder'

csv_df = data_processing.read_csv(
    pd.read_csv(
        os.path.join(ROOT, 'data/metadata/train.csv'),
        index_col=False
    ),
    ROOT
)

# Get dataset and loader
train_dataset = AudioEmotionDataset(csv_df)
train_loader = DataLoader(train_dataset, batch_size=64)

# Initialize the model and the loss function
model = MyModel()  # Substitute with your model
loss_function = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):

        # Forward pass through the model
        outputs = model(inputs)

        # Compute the loss
        loss = loss_function(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```

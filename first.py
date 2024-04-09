import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Load the Daily Dialogue dataset
base_dir = 'ijcnlp_dailydialog'
df = pd.read_csv(os.path.join(base_dir, 'dialogues_text.txt'), sep='\t', names=['utterance'])
emotions = pd.read_csv(os.path.join(base_dir, 'dialogues_emotion.txt'), sep='\t', names=['emotion'])

# Split the emotion strings into lists, skipping rows with invalid values
emotions['emotion'] = emotions['emotion'].apply(lambda x: [int(y) for y in x.split() if y.isdigit()])

# Filter out rows where the emotion list is empty
valid_rows = emotions['emotion'].apply(len) > 0
df = df.loc[valid_rows]
emotions = emotions.loc[valid_rows]

# Ensure the number of utterances and emotions match
# assert len(df) == sum(len(x) for x in emotions['emotion'])

class DailyDialogueDataset(Dataset):
    def __init__(self, df, emotions):
        self.utterances = []
        self.emotions = []
        for i, row in df.iterrows():
            self.utterances.extend(row['utterance'].split(','))
            self.emotions.extend(emotions.iloc[i]['emotion'])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        emotion = self.emotions[idx]
        input_ids = tokenizer.encode(utterance, add_special_tokens=True)
        return torch.tensor(input_ids), torch.tensor(emotion)

dataset = DailyDialogueDataset(df, emotions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the autoencoder attention model
class AutoencoderAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AutoencoderAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.Tanh(),
            nn.Linear(hidden_size // 8, 1),
            nn.Softmax(dim=1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        attention_weights = self.attention(encoded)
        context = torch.sum(encoded * attention_weights, dim=1)
        decoded = self.decoder(context)
        classified = self.classifier(context)
        return decoded, classified


# Train the model
model = AutoencoderAttention(
    input_size=len(tokenizer.get_vocab()), hidden_size=512, num_classes=7
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()

for epoch in range(10):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs, classified = model(inputs)
        loss_rec = criterion(outputs, inputs)
        loss_cls = criterion_cls(classified, labels)
        loss = loss_rec + loss_cls
        loss.backward()
        optimizer.step()

# Cluster the latent representations
with torch.no_grad():
    all_utterances = []
    all_emotions = []
    for inputs, labels in dataloader:
        encoded, _ = model(inputs)
        all_utterances.append(encoded.cpu().numpy())
        all_emotions.append(labels.cpu().numpy())

    all_utterances = np.concatenate(all_utterances, axis=0)
    all_emotions = np.concatenate(all_emotions, axis=0)

    kmeans = KMeans(n_clusters=7, random_state=0)
    clusters = kmeans.fit_predict(all_utterances)
    print("Silhouette score:", silhouette_score(all_utterances, clusters))

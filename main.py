import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import re

# Neural Network for Reward Model
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

# Training and evaluation functions for the Reward Model
def train_reward_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def evaluate_reward_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    print("Accuracy: {:.2f}%".format((1 - avg_loss) * 100))
    return avg_loss

# CSV Handling Functions
def create_csv(file_path, n=20):
    prompts = ["good", "bad", "average", "more technical"]
    csv_dict = {'User': [], 'Prompt': [], 'Mark': [], 'Reward': []}

    def assign_reward(prompt, mark):
        reward_mapping = {
            "good": min(5, mark * 2),
            "bad": max(1, mark - 1),
            "average": mark,
            "more technical": min(5, mark + 1)
        }
        return int(max(1, min(5, reward_mapping.get(prompt, mark))))

    for i in range(n):
        user = f'User {i}'
        current_prompt = random.choice(prompts)
        mark = random.randint(1, 5)
        reward = assign_reward(current_prompt, mark)

        csv_dict['User'].append(user)
        csv_dict['Prompt'].append(current_prompt)
        csv_dict['Mark'].append(mark)
        csv_dict['Reward'].append(reward / 5)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Prompt', 'Mark', 'Reward'])
        for i in range(n):
            writer.writerow([csv_dict['User'][i], csv_dict['Prompt'][i], csv_dict['Mark'][i], csv_dict['Reward'][i]])

def add_line_in_csv(file_path, line_data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line_data)

def read_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

def take_columns(file_path, columns):
    headers, data = read_csv(file_path)
    column_data = {col: [] for col in columns}
    for row in data:
        for col in columns:
            if col in headers:
                index = headers.index(col)
                column_data[col].append(row[index])
    return column_data

# Tokenization and Embedding Functions
def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    unique_tokens = sorted(set(tokens))
    word_to_id = {token: i for i, token in enumerate(unique_tokens)}
    id_to_word = {i: token for token, i in word_to_id.items()}
    return word_to_id, id_to_word

def one_hot_encode(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

def generate_training_data(tokens, word_to_id, window=2):
    X, y = [], []
    vocab_size = len(word_to_id)
    for i, center_word in enumerate(tokens):
        center_index = word_to_id[center_word]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                context_word = tokens[j]
                context_index = word_to_id[context_word]
                X.append(one_hot_encode(center_index, vocab_size))
                y.append(one_hot_encode(context_index, vocab_size))
    return np.array(X), np.array(y)

# Skip-gram Model Functions
def init_network(vocab_size, embedding_dim):
    return {
        "w1": np.random.randn(vocab_size, embedding_dim),
        "w2": np.random.randn(embedding_dim, vocab_size)
    }

def softmax(X):
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(model, X, return_cache=True):
    a1 = X @ model["w1"]
    a2 = a1 @ model["w2"]
    z = softmax(a2)
    if return_cache:
        return {"a1": a1, "a2": a2, "z": z}
    return z

def cross_entropy(z, y):
    return -np.sum(np.log(z + 1e-9) * y)

def backward(model, X, y, alpha):
    cache = forward(model, X)
    dz = cache["z"] - y
    dw2 = cache["a1"].T @ dz
    da1 = dz @ model["w2"].T
    dw1 = X.T @ da1
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

def get_prompt_embedding(prompt, word_to_id, embeddings):
    prompt_embedding = []
    for word in prompt:
        if word in word_to_id:
            embedding = embeddings[word_to_id[word]]
            prompt_embedding.append(embedding)
    return np.mean(prompt_embedding, axis=0) if prompt_embedding else np.zeros(embeddings.shape[1])

# Example Usage
create_csv('data/reward_data.csv', n=20)
add_line_in_csv('data/reward_data.csv', ['User 21', 'good', 5, 1.0])

headers, data = read_csv('data/reward_data.csv')
print("CSV Headers and Data:")
print(headers)
print(data)

datas = take_columns('data/reward_data.csv', ['User', 'Prompt', 'Mark', 'Reward'])
print("\nSelected Columns:")
print(datas)

# Tokenization and Embedding
tokens_nested = [tokenize(prompt) for prompt in datas['Prompt']]
tokens = [word for sublist in tokens_nested for word in sublist]
word_to_id, id_to_word = mapping(tokens)

X, y = generate_training_data(tokens, word_to_id, window=2)

# Training the Skip-gram Model
np.random.seed(42)
vocab_size = len(word_to_id)
embedding_dim = 10
model = init_network(vocab_size, embedding_dim)

history = [backward(model, X, y, alpha=0.05) for _ in range(50)]

plt.plot(range(len(history)), history)
plt.title("Loss over time")
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.show()

# Extracting and Printing Embeddings
embeddings = model["w1"]
prompt_embeddings = [get_prompt_embedding(prompt, word_to_id, embeddings) for prompt in tokens_nested]

print("\nPrompt Embeddings:")
for i, embedding in enumerate(prompt_embeddings):
    print(f"Prompt {i + 1} Embedding: {embedding}")


datas['Prompt Embeddings'] = prompt_embeddings
X_embedding = torch.tensor([l for l in datas['Prompt Embeddings']], dtype=torch.float32)
X_mark = torch.tensor([(int(m) - 1) / 4 for m in datas['Mark']], dtype=torch.float32)
X_reward = torch.tensor(np.array([float(r) for r in datas['Reward']]), dtype=torch.float32)


X = torch.cat((X_embedding, X_mark.unsqueeze(1)), dim=1)
y = X_reward.unsqueeze(1)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Reward Model
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1
reward_model = RewardModel(input_dim, hidden_dim, output_dim)
# Train the reward model
reward_model = train_reward_model(reward_model, [(X_train[i], y_train[i]) for i in range(len(X_train))], num_epochs=10, learning_rate=0.001)
# Evaluate the reward model
evaluate_reward_model(reward_model, [(X_val[i], y_val[i]) for i in range(len(X_val))])
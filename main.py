import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import random

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
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
    return avg_loss


def create_csv(file_path, n=20):
    prompt = ["good", "bad", "average", "more technical"]
    csv_dict = {
        'User': [],
        'Prompt': [],
        'Mark': [],
        'Reward': []  # New column for rewards
    }

    # Define a reward logic based on prompt and mark
    def assign_reward(prompt, mark):
        if prompt == "good":
            reward = min(5, mark * 2)
        elif prompt == "bad":
            reward = max(1, mark - 1)
        elif prompt == "average":
            reward = mark
        elif prompt == "more technical":
            reward = min(5, mark + 1)
        else:
            reward = mark

        # Ensure the reward is an integer between 1 and 5
        reward = int(max(1, min(5, reward)))
        return reward

    for i in range(n):
        user = f'User {i}'
        current_prompt = random.choice(prompt)
        mark = random.randint(1, 5)
        reward = assign_reward(current_prompt, mark)

        csv_dict['User'].append(user)
        csv_dict['Prompt'].append(current_prompt)
        csv_dict['Mark'].append(mark)
        csv_dict['Reward'].append(reward / 5)  # Normalize by dividing by 5

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
    data = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read header row
        for row in reader:
            data.append(row)
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

# Example usage
create_csv('data/reward_data.csv', n=20)
add_line_in_csv('data/reward_data.csv', ['User 21', 'good', 5, 1.0])
print()
print(read_csv('data/reward_data.csv'))
print()
print("\n".join("{}\t{}".format(k, v) for k, v in take_columns('data/reward_data.csv', ['User', 'Mark', 'Reward']).items()))
#!usr/bin/env python

import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""

abbreviation key

gp = games played

net_rating = offRating - defRating

offRating = 100*((points)/POSS)

defRating = 100*((opp points/(opp POSS)))

oreb_pct = offensive rebound percentage

usg_pct = usage percentage is a measurement of the percentage of team plays utilized by a player while they are in the game

ts_pct = true shooting percentage.  percentage of shots made factoring in threes and free throws.  

ast_ptg = assist percentage.  Percent of field goals (2 or 3 point shots not including free throws) 

"""

csv = pd.read_csv("all_seasons.csv")
total_players = len(csv["player_name"])
print(csv.columns, total_players)

colleges = []
countries = []
years = []
seasons = []
teams = []
for i in range(0, total_players):
    if csv["college"][i] not in colleges:
        colleges.append(csv["college"][i])
    if csv["country"][i] not in countries:
        countries.append(csv["country"][i])
    if csv["draft_year"][i] not in colleges:
        years.append(csv["draft_year"][i])
    if csv["season"][i] not in seasons:
        seasons.append(csv["season"][i])
    if csv["team_abbreviation"][i] not in teams:
        teams.append(csv["team_abbreviation"][i])
print(colleges, countries, years, seasons, teams)

sum_round = 0
count_round = 0
sum_number = 0
count_number = 0
for i in range(0, total_players):
    if csv["draft_number"][i] != "Undrafted":
        sum_number += int(csv["draft_number"][i])
        count_number += 1
    if csv["draft_round"][i] != "Undrafted":
        sum_round += int(csv["draft_round"][i])
        count_round += 1

mean_round = sum_round / count_round
mean_number = sum_number / count_number
print(mean_round, mean_number)

for i in range(0, total_players):
    if csv["draft_number"][i] == "Undrafted":
        csv["draft_number"][i] = mean_number
    if csv["draft_round"][i] == "Undrafted":
        csv["draft_round"][i] = mean_round


def one_hot(value, array):
    v = [0 for i in range(len(array))]
    v[array.index(value)] = 1
    return torch.Tensor(v)


# Data Preprocessing


def random_split(test_percent):
    test_amount = int(total_players * test_percent)
    test_sample = random.sample(range(0, total_players), test_amount)
    train_sample = [i for i in range(0, total_players)]
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    for idx in test_sample:
        train_sample.remove(idx)
        test_x.append(
            [
                one_hot(csv["team_abbreviation"][idx], teams),
                one_hot(csv["college"][idx], colleges),
                one_hot(csv["country"][idx], countries),
                one_hot(csv["draft_year"][idx], years),
                one_hot(csv["season"][idx], seasons),
                float(csv["age"][idx]),
                float(csv["player_height"][idx]),
                float(csv["player_weight"][idx]),
                float(csv["draft_round"][idx]),
                float(csv["draft_number"][idx]),
                float(csv["gp"][idx]),
                float(csv["net_rating"][idx]),
                float(csv["usg_pct"][idx]),
            ]
        )
        test_y.append(
            torch.Tensor(
                [
                    float(csv["pts"][idx]),
                    float(csv["reb"][idx]),
                    float(csv["ast"][idx]),
                    float(csv["oreb_pct"][idx]),
                    float(csv["dreb_pct"][idx]),
                    float(csv["ts_pct"][idx]),
                    float(csv["ast_pct"][idx]),
                ]
            )
        )
    for idx in train_sample:
        train_x.append(
            [
                one_hot(csv["team_abbreviation"][idx], teams),
                one_hot(csv["college"][idx], colleges),
                one_hot(csv["country"][idx], countries),
                one_hot(csv["draft_year"][idx], years),
                one_hot(csv["season"][idx], seasons),
                float(csv["age"][idx]),
                float(csv["player_height"][idx]),
                float(csv["player_weight"][idx]),
                float(csv["draft_round"][idx]),
                float(csv["draft_number"][idx]),
                float(csv["gp"][idx]),
                float(csv["net_rating"][idx]),
                float(csv["usg_pct"][idx]),
            ]
        )
        train_y.append(
            torch.Tensor(
                [
                    float(csv["pts"][idx]),
                    float(csv["reb"][idx]),
                    float(csv["ast"][idx]),
                    float(csv["oreb_pct"][idx]),
                    float(csv["dreb_pct"][idx]),
                    float(csv["ts_pct"][idx]),
                    float(csv["ast_pct"][idx]),
                ]
            )
        )
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = random_split(0.2)

# Training function


def train(
    neural_net,
    optimizer,
    loss,
    scheduler,
    train_features,
    train_labels,
    epochs,
    batch_size,
    dropout=False,
):
    xs = [[] for i in range(epochs)]
    ys = [[] for i in range(epochs)]
    neural_net.train()
    for epoch in range(epochs):
        count = 0
        rl = 0.0
        optimizer.zero_grad()
        for i in range(len(train_labels)):
            train_y = train_labels[i]
            train_x = train_features[i]
            output = neural_net(train_x)
            out_loss = loss(output, train_y)
            out_loss.backward()
            optimizer.step()
            count += 1
            rl += out_loss.item()
            if count % batch_size == 0:
                optimizer.zero_grad()
                print(str(count) + " completed. Loss: " + str(rl / batch_size))
                xs[epoch].append(count)
                ys[epoch].append(rl / batch_size)
                rl = 0.0
        scheduler.step()
    return xs, ys


# Neural net model declaration


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.teamLin = nn.Linear(len(teams), 1)
        self.collegeLin = nn.Linear(len(colleges), 1)
        self.countryLin = nn.Linear(len(countries), 1)
        self.draftLin = nn.Linear(len(years), 1)
        self.seasonLin = nn.Linear(len(seasons), 1)
        self.sequential = nn.Sequential(
            nn.Linear(13, 20),
            # nn.Dropout(p=0.2),
            nn.Linear(20, 40),
            # nn.Dropout(p=0.2),
            nn.Linear(40, 40),
            # nn.Dropout(p=0.2),
            nn.Linear(40, 40),
            # nn.Dropout(p=0.2),
            nn.Linear(40, 20),
            # nn.Dropout(p=0.2),
            nn.Linear(20, 7),
        )

    def forward(self, x):
        x = torch.Tensor(
            [
                self.teamLin(x[0]),
                self.collegeLin(x[1]),
                self.countryLin(x[2]),
                self.draftLin(x[3]),
                self.seasonLin(x[4]),
            ]
            + x[5:]
        )
        return self.sequential(x)


# Linear approximation model declaration


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.teamLin = nn.Linear(len(teams), 1)
        self.collegeLin = nn.Linear(len(colleges), 1)
        self.countryLin = nn.Linear(len(countries), 1)
        self.draftLin = nn.Linear(len(years), 1)
        self.seasonLin = nn.Linear(len(seasons), 1)
        self.sequential = nn.Sequential(nn.Linear(13, 7))

    def forward(self, x):
        x = torch.Tensor(
            [
                self.teamLin(x[0]),
                self.collegeLin(x[1]),
                self.countryLin(x[2]),
                self.draftLin(x[3]),
                self.seasonLin(x[4]),
            ]
            + x[5:]
        )
        return self.sequential(x)


# Initialization of objects necessary


neural = NeuralNet()
linnet = LinearNet()
loss = nn.MSELoss()
lin_loss = nn.MSELoss()
lin_optim = optim.Adam(linnet.parameters())
optimizer = optim.Adam(neural.parameters())
n_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

xs, ys = train(neural, optimizer, loss, n_scheduler, train_x, train_y, 30, 200)

for i in range(len(xs)):
    plt.plot(xs[i], ys[i])

l = []
x = []
for i in range(len(test_x)):
    l.append(loss(neural(test_x[i]), test_y[i]).item())
    x.append(i)
plt.hist(l, range(0, 25))

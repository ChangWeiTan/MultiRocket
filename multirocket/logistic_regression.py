import copy

import numpy as np
import torch
import torch.nn.functional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class LogisticRegression:

    def __init__(
            self,
            num_features,
            max_epochs=500,
            minibatch_size=256,
            validation_size=2 ** 11,
            learning_rate=1e-4,
            patience_lr=5,  # 50 minibatches
            patience=10,  # 100 minibatches
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.name = "LogisticRegression"
        self.args = {
            "num_features": num_features,
            "validation_size": validation_size,
            "minibatch_size": minibatch_size,
            "lr": learning_rate,
            "max_epochs": max_epochs,
            "patience_lr": patience_lr,
            "patience": patience,
        }

        self.model = None
        self.device = device
        self.classes = None
        self.scaler = None
        self.num_classes = None

    def fit(self, x_train, y_train):
        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        num_outputs = self.num_classes if self.num_classes > 2 else 1
        train_steps = int(x_train.shape[0] / self.args["minibatch_size"])

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)

        model = torch.nn.Sequential(torch.nn.Linear(self.args["num_features"], num_outputs)).to(self.device)

        if num_outputs == 1:
            loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            min_lr=1e-8,
            patience=self.args["patience_lr"]
        )

        training_size = x_train.shape[0]
        if self.args["validation_size"] < training_size:
            x_training, x_validation, y_training, y_validation = train_test_split(
                x_train, y_train,
                test_size=self.args["validation_size"],
                stratify=y_train
            )

            train_data = TensorDataset(
                torch.tensor(x_training, dtype=torch.float32).to(self.device),
                torch.tensor(y_training, dtype=torch.long).to(self.device)
            )
            val_data = TensorDataset(
                torch.tensor(x_validation, dtype=torch.float32).to(self.device),
                torch.tensor(y_validation, dtype=torch.long).to(self.device)
            )
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"])
            val_dataloader = DataLoader(val_data, batch_size=self.args["minibatch_size"])
        else:
            train_data = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32).to(self.device),
                torch.tensor(y_train, dtype=torch.long).to(self.device)
            )
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"])
            val_dataloader = None

        best_loss = np.inf
        best_model = None
        stall_count = 0
        stop = False

        for epoch in range(self.args["max_epochs"]):
            if epoch > 0 and stop:
                break
            model.train()

            # loop over the training set
            total_train_loss = 0
            steps = 0
            for i, data in tqdm(enumerate(train_dataloader), desc=f"epoch: {epoch}", total=train_steps):
                x, y = data

                y_hat = model(x)
                if num_outputs == 1:
                    loss = loss_function(y_hat.sigmoid(), y)
                else:
                    yhat = torch.nn.functional.softmax(y_hat, dim=1)
                    loss = loss_function(yhat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss
                steps += 1

            total_train_loss = total_train_loss.cpu().detach().numpy() / steps

            if val_dataloader is not None:
                total_val_loss = 0
                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    model.eval()
                    for i, data in enumerate(val_dataloader):
                        x, y = data

                        y_hat = model(x)
                        if num_outputs == 1:
                            total_val_loss += loss_function(y_hat.sigmoid(), y)
                        else:
                            yhat = torch.nn.functional.softmax(y_hat, dim=1)
                            total_val_loss += loss_function(yhat, y)
                total_val_loss = total_val_loss.cpu().detach().numpy() / steps
                scheduler.step(total_val_loss)

                if total_val_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_val_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0
            else:
                scheduler.step(total_train_loss)
                if total_train_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_train_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0

        self.model = best_model
        return self.model

    def predict(self, x):
        x = self.scaler.transform(x)

        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()

            yhat = self.model(torch.tensor(x, dtype=torch.float32).to(self.device))

            if self.num_classes > 2:
                yhat = self.classes[np.argmax(yhat.cpu().detach().numpy(), axis=1)]
            else:
                yhat = torch.sigmoid(yhat)
                yhat = np.round(yhat.cpu().detach().numpy())

            return yhat

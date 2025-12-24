import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self,model,train_loader,val_loader,device,config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        if config["optimizer"]["type"].lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config["optimizer"]["lr"],
                momentum=config["optimizer"]["momentum"],
                weight_decay=config["optimizer"]["weight_decay"]
            )
        elif config["optimizer"]["type"].lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config["optimizer"]["type"]}")
        
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"]
        )
        
        self.save_dir = config["train"]["save_dir"]
        os.makedirs(self.save_dir,exist_ok=True)

    def train_epoch(self,epoch):
        self.model.train()
        total_loss,correct,total = 0.0,0,0
        
        loop = tqdm(self.train_loader,desc=f"Epochs{epoch}")
        for images,labels in loop:
            images,labels = images.to(self.device),labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _,preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(),acc=100.*correct/total)
        
        self.scheduler.step()
        avg_loss = total_loss/len(self.train_loader)
        acc = 100.*correct/total
        return avg_loss,acc
    
        
    def val(self):

        self.model.eval()
        total_loss,correct,total = 0.0,0,0

        with torch.no_grad():
            for images,labels in self.val_loader:
                images,labels = images.to(self.device),labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs,labels)

                total_loss += loss.item()
                _,preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
            
        avg_loss = total_loss/len(self.val_loader)
        acc = 100.*correct/total

        return avg_loss,acc
    
    def save_checkpoints(self,epoch,acc):
        path = os.path.join(self.save_dir,f"Epoch{epoch}_acc_{acc:.2f}.pt")
        torch.save(self.model.state_dict(),path)
        print(f"Checkpoints saved at path :{path}")
    
    def fit(self):
        best_acc = 0
        for epoch in range(1,self.config["train"]["epochs"] + 1):
            train_loss,train_acc = self.train_epoch(epoch)
            val_loss,val_acc = self.val()

            print(f"\nEpoch {epoch}:"
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}   |"
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}"   )

            if val_acc>best_acc:
                best_acc = val_acc
                self.save_checkpoints(epoch,val_acc)       
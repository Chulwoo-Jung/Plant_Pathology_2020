import torch 
from tqdm import tqdm 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, 
                 scheduler, device, metric, cutmix_prob):
        self.model = model.to(device)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = optimizer 
        self.train_loader = train_loader 
        self.val_loader = val_loader 
        self.scheduler = scheduler 
        self.device = device 
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.metric = metric
        self.cutmix_prob = cutmix_prob

    def train_epoch(self, epoch):
        self.model.train()

        accu_loss = 0.0
        running_avg_loss = 0.0
        self.metric.reset()

        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1} [Training..]') as pbar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Apply mixup
                if self.cutmix_prob > 0 and np.random.rand(1) < self.cutmix_prob:
                    inputs, targets, shuffled_targets, lam = self.cutmix(inputs, targets.clone())
                    outputs = self.model(inputs)
                    loss = lam * self.loss_fn(outputs, targets) + (1 - lam) * self.loss_fn(outputs, shuffled_targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)


                # Backward Pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accu_loss += loss.item()
                running_avg_loss = accu_loss / (batch_idx + 1)

                pred_proba = F.softmax(outputs, dim=1)
                self.metric.update(pred_proba, targets)

                pbar.update(1)
                if batch_idx % 20 == 0 or (batch_idx + 1) == len(self.train_loader):
                    pbar.set_postfix({
                        'loss': running_avg_loss,
                        'metric': self.metric.compute().item()
                    })

        return running_avg_loss, self.metric.compute().item()
    
    def val_epoch(self, epoch):
        if not self.val_loader:
            return None, None
          
        self.model.eval()

        accu_loss = 0.0
        running_avg_loss = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        self.metric.reset()

        with tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1} [Validation..]') as pbar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Forward Pass
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                    accu_loss += loss.item()
                    running_avg_loss = accu_loss / (batch_idx + 1)

                    pred_proba = F.softmax(outputs, dim=1)
                    self.metric.update(pred_proba, targets)

                    pbar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == len(self.val_loader):
                        pbar.set_postfix({
                            'loss': running_avg_loss,
                            'metric': self.metric.compute().item()
                        })
        
        self.scheduler.step(running_avg_loss)
        self.current_lr = self.optimizer.param_groups[0]['lr']

        return running_avg_loss, self.metric.compute().item()

    def fit(self, epochs):
        history = {
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': [],
            'lr': []
        }

        for epoch in range(epochs):
            train_loss, train_metric = self.train_epoch(epoch)
            val_loss, val_metric = self.val_epoch(epoch)
            
            if val_loss is not None and val_metric is not None:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}, Lr: {self.current_lr:.6f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, '
                      f'Lr: {self.current_lr:.6f}')

            history['train_loss'].append(train_loss)
            history['train_metric'].append(train_metric)
            if val_loss is not None and val_metric is not None:
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)
            history['lr'].append(self.current_lr)

        return history 
        
    def get_trained_model(self):
        return self.model 
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.array(W * cut_rat).astype(np.int32)
        cut_h = np.array(H * cut_rat).astype(np.int32)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)

        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def cutmix(self, images, targets):
        indicies = torch.randperm(images.size(0))
        shuffled_images = images[indicies]
        shuffled_targets = targets[indicies]
        
        lam = np.random.beta(1.0, 1.0)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[ :, :, bbx1:bbx2, bby1:bby2] = shuffled_images[ :, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return images, targets, shuffled_targets, lam        
        
        
        

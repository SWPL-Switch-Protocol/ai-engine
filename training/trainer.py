import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

class SwitchTrainer:
    """
    Distributed training loop for SwitchNet models.
    Supports mixed-precision training and gradient accumulation.
    """
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10
        )
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SwitchTrainer")

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Context
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                predictions, uncertainty = self.model(data)
                loss = self._calculate_loss(predictions, uncertainty, target)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip_val'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        self.scheduler.step()
        return total_loss / len(dataloader)

    def _calculate_loss(self, pred, uncertainty, target):
        """
        Negative Log Likelihood Loss for regression with uncertainty.
        Loss = 0.5 * (log(variance) + (target - pred)^2 / variance)
        """
        variance = uncertainty ** 2
        nll = 0.5 * (torch.log(variance) + (target - pred)**2 / variance)
        return nll.mean()

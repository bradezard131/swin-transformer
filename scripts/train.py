from fastprogress.fastprogress import master_bar, progress_bar
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as D, transforms as T
import wandb

import swin


@hydra.main(config_path='config', config_name='default')
def main(
    cfg: DictConfig,
):
    assert torch.cuda.is_available() or 'cuda' not in cfg.device, "Cannot select CUDA when no CUDA devices are available"
    print(f'Training using device {cfg.device}')
    device = torch.device(cfg.device)
    
    if cfg.dataset.name.lower() == 'cifar10':
        print(f'Training on CIFAR10')
        train_dataset = D.CIFAR10(
            cfg.dataset.root,
            train=True,
            transform=T.Compose([
                T.Resize(128),
                T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            download=True
        )
        
        val_dataset = D.CIFAR10(
            cfg.dataset.root,
            train=True,
            transform=T.Compose([
                T.Resize(128),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            download=True
        )
    else:
        raise NotImplemented
    
    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.dataloader.num_workers,
    )
    
    val_dl = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        pin_memory=True,
        num_workers=cfg.dataloader.num_workers,
    )
    
    print(f'Building Swin model')
    model = swin.build_swin_model(cfg.dataset.num_classes, **cfg.model).to(device)
    
    if cfg.optim.name.lower() == 'sgd':
        print(f'Building SGD optimizer')
        optim = torch.optim.SGD(
            model.parameters(),
            lr = cfg.optim.lr,
            momentum = cfg.optim.momentum,
            weight_decay = cfg.optim.weight_decay,
        )
    else:
        raise NotImplemented
    
    wandb.init(config=cfg)  # type: ignore
    wandb.watch(model, log_freq=100)
    
    epoch_bar = master_bar(range(cfg.epochs))
    for epoch in epoch_bar:
        model.train()
        for i, (batch, labels) in enumerate(progress_bar(train_dl, parent=epoch_bar)):
            loss = F.cross_entropy(model(batch.to(device)), labels.to(device))
            loss.backward()
            optim.step()
            optim.zero_grad(True)
            
            if i % cfg.log_interval == 0:
                float_loss = loss.item()
                epoch_bar.child.comment = f'Loss: {float_loss:7.04f}'
                wandb.log({'loss': float_loss})
        
        model.eval()
        correct = 0
        with torch.inference_mode():
            for batch, labels in progress_bar(val_dl, parent=epoch_bar):
                preds = model(batch.to(device)).argmax(-1)
                correct += (preds == labels.to(device)).sum().item()
        epoch_bar.write(f'Epoch {epoch:d}: Accuracy = {100 * correct / len(val_dataset):5.02f}%')


if __name__ == '__main__':
    main()
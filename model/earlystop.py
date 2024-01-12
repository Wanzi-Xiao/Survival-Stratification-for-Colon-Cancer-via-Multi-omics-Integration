import torch

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        :param patience (int): How long to wait after last time validation loss improved.
                               Default: 7
        :param min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                  Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        :param val_loss: Current validation loss.
        :param model: Model to save.
        """
        if val_loss < self.val_loss_min:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss

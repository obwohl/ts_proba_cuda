import copy
import os

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, verbose=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == 'cosine_warmup':
        warmup_epochs = getattr(args, 'warmup_epochs', 10)
        if epoch <= warmup_epochs:
            # Linearer Warmup
            # Verhindert Division durch Null, wenn warmup_epochs=0 ist.
            if warmup_epochs > 0:
                lr = args.lr * (epoch / warmup_epochs)
            else:
                # Wenn kein Warmup, sollte dieser Zweig für Epoche >= 1 nicht erreicht werden.
                # Zur Sicherheit wird die LR auf den Startwert gesetzt.
                lr = args.lr
        else:
            # Cosine Annealing
            decay_epochs = args.num_epochs - warmup_epochs
            # Stelle sicher, dass wir eine Decay-Phase haben
            if decay_epochs > 0:
                # Korrekte Berechnung des Fortschritts (von 0 bis 1) innerhalb der Decay-Phase
                current_decay_epoch = epoch - warmup_epochs
                progress = (current_decay_epoch - 1) / (decay_epochs - 1) if decay_epochs > 1 else 1.0
                lr = 0.5 * args.lr * (1. + math.cos(math.pi * progress))
            else:
                # Wenn keine Decay-Phase nach dem Warmup übrig ist, halte die LR konstant.
                lr = args.lr
        lr_adjust = {epoch: lr}
    elif args.lradj == 'plateau':
        return
    else:
        lr_adjust = {}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if verbose:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Eine Verbesserung wird nur gezählt, wenn `neuer_loss <= alter_loss - delta`.
                            Default: 0
            path (str):     Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path


    def __call__(self, val_loss, model_state):
        # --- START OF FIX: Robust handling of NaN/Inf ---
        # 1. Check for invalid loss values (NaN or Inf).
        #    These are always treated as a non-improvement.
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.counter += 1
            if self.verbose:
                # Print a clear message why the counter is being incremented.
                print(f"Invalid validation loss ({val_loss}). EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return  # Exit the function here to avoid the faulty logic below.

        # 2. The remaining logic is only executed for valid, finite loss values.
        # Check if the current loss is a significant improvement.
        if val_loss < self.val_loss_min - self.delta:
            # This is an improvement. Save the model and reset the counter.
            self.save_checkpoint(val_loss, model_state)
            self.counter = 0
        else:
            # No significant improvement. Increment the counter.
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # --- END OF FIX ---

    def save_checkpoint(self, val_loss, model_state):
        '''Saves model to file when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        try:
            checkpoint_dir = os.path.dirname(self.path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

            absolute_path = os.path.abspath(self.path)
            print(f"---&gt; Attempting to save model to: {absolute_path}")
            torch.save(model_state, self.path)
            print(f"---&gt; Model successfully saved.")
            self.val_loss_min = val_loss
        except Exception as e:
            print(f"!!!!!! FAILED TO SAVE MODEL to path: {self.path} !!!!!!")
            print(f"!!!!!! Error: {e}")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

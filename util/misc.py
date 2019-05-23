import torch
from torch import nn
import torch.optim.lr_scheduler
from torch.utils.data import WeightedRandomSampler

from models.classifier import Classifier
from models.loss import FocalLoss, FBetaLoss

import numpy as np
from sklearn.metrics import confusion_matrix

def get_model(args):
	return Classifier(args.arch)

def get_loss(args):

	if args.loss == "bce":
		loss_func = nn.BCEWithLogitsLoss()
	elif args.loss == "focal":
		if "focal_gamma" in args.loss_params:
			loss_func = FocalLoss(gamma=args.loss_params["focal_gamma"])
		else:
			loss_func = FocalLoss()
	elif args.loss == "fbeta":
		if "fbeta" in args.loss_params:
			loss_func = FBetaLoss(beta=args.loss_params["fbeta"], soft=True)
		else:
			loss_func = FBetaLoss(soft=True)
	elif args.loss == "softmargin":
		loss_func = nn.SoftMarginLoss()
	else:
		raise ValueError(f"Invalid loss function specifier: {args.loss}")

	return loss_func

def get_train_sampler(args, dataset):
	if args.example_weighting:
		return WeightedRandomSampler(weights=dataset.example_weights, num_samples=len(dataset))
	else:
		return None

def get_scheduler(args, optimizer):
	params = args.lr_schedule_params
	if args.lr_schedule == "poly":
		gamma = params["gamma"] if "gamma" in params else 0.9
		max_iter = args.epochs
		decay_iter = 1
		return PolynomialLR(optimizer, max_iter, decay_iter, gamma)
	elif args.lr_schedule == "exp":
		gamma = params["gamma"] if "gamma" in params else 0.95
		return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
	elif args.lr_schedule == "step":
		step_size = params["step_size"] if "step_size" in params else 5
		gamma = params["gamma"] if "gamma" in params else 0.5
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
	elif args.lr_schedule == "multistep":
		milestones = params["milestones"] if "milestones" in params else list(range(10, args.epochs, 10))
		gamma = params["gamma"] if "gamma" in params else 0.2
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
	elif args.lr_schedule == "cosine":
		T_max = params["period"] // 2 if "period" in params else 10
		max_decay = params["max_decay"] if "max_decay" in params else 50
		eta_min = args.initial_lr / max_decay
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
	else:
		return ConstantLR(optimizer)

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
		self.decay_iter = decay_iter
		self.max_iter = max_iter
		self.gamma = gamma
		super(PolynomialLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
			return [base_lr for base_lr in self.base_lrs]
		else:
			factor = (1 - (self.last_epoch / self.max_iter)) ** self.gamma
			return [base_lr * factor for base_lr in self.base_lrs] 

class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, last_epoch=-1):
		super(ConstantLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr for base_lr in self.base_lrs]

def sensitivity_specificity(y_true, y_pred):

	cfm = confusion_matrix(y_true, y_pred, labels=[0,1])

	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	# Sensitivity, hit rate, recall, or true positive rate
	tpr = tp/(tp+fn)

	# Specificity or true negative rate
	tnr = tn/(tn+fp)

	# Precision or positive predictive value
	ppv = tp/(tp+fp)

	# Negative predictive value
	npv = tn/(tn+fn)

	# Fall out or false positive rate
	fpr = fp/(fp+tn)

	# False negative rate
	fnr = fn/(tp+fn)

	# False discovery rate
	fdr = fp/(tp+fp)

	# Overall accuracy
	acc = (tp+tn)/(tp+fp+fn+tn)

	return tpr, tnr

import os
import datetime
import pickle
import json
import csv
import shutil
import torch
import numpy as np
import pandas as pd
import subprocess

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


class Logger:

	def __init__(self, path=None):

		if path is not None:
			if not os.path.isdir(path):
				raise ValueError("Invalid logger path given.")
			self.path = path
			self.dt = path.replace('/','').replace('.','').replace('saves','')
			self.main_log_fn = os.path.join(self.path, "test.txt")

		else:
			self.dt = datetime.datetime.now().strftime("%m%d_%H%M")
			self.path = "./saves/{}".format(self.dt)
			if not os.path.exists(self.path):
				os.makedirs(self.path)
			self.losses = []
			self.scores = []
			self.eval_metrics = set()
			self.main_log_fn = os.path.join(self.path, "log.txt")
			shutil.copy2("args.py", self.path)

	def save_model(self, model, epoch):
		if isinstance(epoch, int):
			fn = os.path.join(self.path, "save_{:03d}.pth".format(epoch))
		else:
			fn = os.path.join(self.path, "save_{}.pth".format(epoch))
		torch.save(model.state_dict(), fn)
		self.print("Saved model to: {}\n".format(fn))

	def print(self, *x):
		print(*x)
		self.log(*x)

	def log(self, *x):
		with open(self.main_log_fn, "a") as f:
			print(*x, file=f, flush=True)

	def log_loss(self, l):
		self.losses.append(l)

	def log_eval(self, data):
		self.eval_metrics = set.union(self.eval_metrics, set(data.keys()))
		for k in self.eval_metrics:
			if not k in data:
				data[k] = ''
		self.scores.append(data)

	def run_test(self, epoch):
		cmd = ["python3", "test.py", self.dt, epoch, "False"]
		self.print(" ".join(cmd))
		subprocess.run(cmd, shell=False)

	def save(self):

		with open(os.path.join(self.path, "loss.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it", "loss"])
			csvwriter.writeheader()
			for it, loss in enumerate(self.losses):
				row = {"it": it, "loss": loss}
				csvwriter.writerow(row)

		train_metrics = [s.split('-')[-1] for s in self.eval_metrics if "train" in s]
		val_metrics   = [s.split('-')[-1] for s in self.eval_metrics if "val" in s]

		with open(os.path.join(self.path, "train_eval.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it"] + sorted(train_metrics))
			csvwriter.writeheader()
			for it, score in enumerate(self.scores):
				row = {k.split('-')[-1]:v for k,v in score.items() if "train" in k}
				if not row: continue
				row["it"] = it
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "eval.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it"] + sorted(val_metrics))
			csvwriter.writeheader()
			for it, score in enumerate(self.scores):
				row = {k.split('-')[-1]:v for k,v in score.items() if "val" in k}
				if not row: continue
				row["it"] = it
				csvwriter.writerow(row)

		plt.clf()

		loss_data = pd.read_csv(os.path.join(self.path, "loss.csv"))
		loss_means = loss_data.copy()
		loss_means.loss = loss_means.loss.rolling(200, center=True, min_periods=1).mean()
		lossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = loss_data,
			color = "b"
		)
		lossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = loss_means,
			color = "orange"
		)
		lossplot.set_title("Train loss")
		plt.ylim(0, 1)
		lossplot.figure.savefig(os.path.join(self.path, "train_loss.png"))

		plt.clf()
		plt.close()

		train_eval_data = pd.read_csv(os.path.join(self.path, "train_eval.csv"))
		evalplot = train_eval_data.plot(x="it", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = train_eval_data.plot(x="it", y="f1",  legend=False, color="r", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="acc", legend=False, color="g", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="sensitivity", legend=False, color="m", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="specificity", legend=False, color="c", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Train Set")
		evalplot.figure.savefig(os.path.join(self.path, "train_eval.png"))
		plt.clf()
		plt.close()

		eval_data = pd.read_csv(os.path.join(self.path, "eval.csv"))

		evalplot = eval_data.plot(x="it", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = eval_data.plot(x="it", y="f1",   legend=False, color="r", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="acc",  legend=False, color="g", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="sensitivity", legend=False, color="m", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="specificity", legend=False, color="c", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Validation Set")
		evalplot.figure.savefig(os.path.join(self.path, "eval.png"))

		plt.clf()
		plt.close()

		f1plot = sns.lineplot(
			x = "it",
			y = "f1",
			data = eval_data
		)
		f1plot.set_title("Eval F1 Score")
		f1plot.figure.savefig(os.path.join(self.path, "eval_f1.png"))

		plt.clf()
		plt.close()

		sensplot = sns.lineplot(
			x = "it",
			y = "sensitivity",
			data = eval_data
		)
		sensplot.set_title("Eval Sensitivity Score")
		sensplot.figure.savefig(os.path.join(self.path, "eval_sensitivity.png"))

		plt.clf()
		plt.close()

		specplot = sns.lineplot(
			x = "it",
			y = "specificity",
			data = eval_data
		)
		specplot.set_title("Eval Specificity Score")
		specplot.figure.savefig(os.path.join(self.path, "eval_specificity.png"))

		plt.clf()
		plt.close()

		accplot = sns.lineplot(
			x = "it",
			y = "acc",
			data = eval_data
		)
		accplot.set_title("Eval Accuracy")
		accplot.figure.savefig(os.path.join(self.path, "eval_acc.png"))

		plt.clf()
		plt.close()

		evallossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = eval_data
		)
		evallossplot.set_title("Eval Loss")
		evallossplot.figure.savefig(os.path.join(self.path, "eval_loss.png"))

		plt.clf()
		plt.close()

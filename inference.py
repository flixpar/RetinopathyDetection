import os
import tqdm
import argparse
import numpy as np
import importlib.util
from collections import OrderedDict
import datetime
import itertools
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from loaders.inference_loader import RetinaImageInferenceDataset
from models.classifier import Classifier

from util.misc import get_model

primary_device = torch.device("cuda:0")

def main(cfg):

	folder_path = os.path.join("./saves", cfg.folder_name)
	if not os.path.exists(folder_path):
		raise ValueError(f"No matching save folder: {folder_path}")

	if cfg.save_id.isdigit() and os.path.exists(os.path.join(folder_path, f"save_{int(cfg.save_id):03d}.pth")):
		save_path = os.path.join(folder_path, f"save_{int(cfg.save_id):03d}.pth")
	elif os.path.exists(os.path.join(folder_path, f"save_{cfg.save_id}.pth")):
		save_path = os.path.join(folder_path, f"save_{cfg.save_id}.pth")
	else:
		raise Exception(f"Specified save not found: {cfg.save_id}")

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()

	test_dataset = RetinaImageInferenceDataset(split=cfg.split, args=args, test_transforms=args.test_augmentation, debug=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size//2, num_workers=args.workers, pin_memory=True)

	model = get_model(args)
	state_dict = torch.load(save_path)
	if "module." == list(state_dict.keys())[0][:7]:
		temp_state = OrderedDict()
		for k, v in state_dict.items():
			temp_state[k.replace("module.", "", 1)] = v
		state_dict = temp_state
	model.load_state_dict(state_dict)
	model.to(primary_device)

	print("Test")
	evaluate(model, test_loader, folder_path, cfg)

def evaluate(model, loader, folder_path, cfg):
	model.eval()

	preds = []
	persons = []
	image_nums = []

	threshold = cfg.thresh

	with torch.no_grad():
		for i, (images, person, image_num) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			if len(images.shape) == 5:
				n_examples, n_copies, c, w, h = images.shape
				images = images.view(n_examples*n_copies, c, w, h)
			else:
				n_examples, _, _, _ = images.shape
				n_copies = 1

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True)

			_, output = model(images)

			if n_copies != 1:
				output = torch.chunk(output, chunks=n_examples, dim=0)
				output = torch.stack(output, dim=0).squeeze(-1)
				output = (0.5 * output[:, 0]) + (0.5 * output[:, 1:].mean(dim=1))

			pred = output.cpu().numpy().squeeze().tolist()
			if not isinstance(pred, list): pred = [pred]
			preds.extend(pred)

			person = person.cpu().numpy().astype(np.int).squeeze().tolist()
			if not isinstance(person, list): person = [person]
			persons.extend(person)

			image_num = image_num.cpu().numpy().astype(np.int).squeeze().tolist()
			if not isinstance(image_num, list): image_num = [image_num]
			image_nums.extend(image_num)

	scores = np.array(preds).squeeze()
	persons = np.array(persons).squeeze()
	image_nums = np.array(image_nums).squeeze()

	dt = datetime.datetime.now().strftime("%m%d%H%M")
	test_folder_path = os.path.join(folder_path, f"inference_{dt}")
	if not os.path.isdir(test_folder_path): os.makedirs(test_folder_path)

	preds = scores > threshold
	preds = preds.astype(np.int)

	with open(os.path.join(test_folder_path, "cfg.txt"), "w") as f:
		f.write(f"folder:    {cfg.folder_name}\n")
		f.write(f"epoch:     {cfg.save_id}\n")
		f.write(f"dataset:   {cfg.split}\n")
		f.write(f"threshold: {threshold}\n")

	results = pd.DataFrame({
		"person": persons,
		"image":  image_nums,
		"pred":   preds,
                "score":  scores,
	})
	results.to_csv(os.path.join(test_folder_path, "preds.csv"), index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Test for Retina Project")
	parser.add_argument("folder_name", type=str,   help="Name of save folder")
	parser.add_argument("save_id",     type=str,   help="Name of save epoch")
	parser.add_argument("--split",     type=str,   required=False, default="test", help="Dataset partition")
	parser.add_argument("--thresh",    type=float, required=False, default=0.5,    help="Prediction threshold")
	args = parser.parse_args()
	main(args)

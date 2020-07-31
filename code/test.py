from tabulate import tabulate

import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
from train import get_ext_parser, model_dict

parser = get_ext_parser()
parser.add_argument('--input_theta', type = float, default = -1)

if __name__ == "__main__":
	args = parser.parse_args()

	if args.model_name == models.TREX.name:
		con = config.TopicAwareConfig(args)
	else:
		con = config.PairwiseConfig(args)

	con.load_test_data()

	model_pattern = model_dict[args.model_name]
	if args.save_name:
		save_name_base = args.save_name
	else:
		save_name_base = con.get_expname(model_pattern)

	pretrain_model = f"{save_name_base}_best"

	model, optimizer = con.load_model_optimizer(model_pattern, pretrain_model)
	save_name = f"{save_name_base}"
	model.eval()

	print("Dev ==============================================")
	#f1, auc, pr_x, pr_y, theta, theta_ign, result_dict = con.test(model)
	res_dict_dev = con.full_test(model)
	columns = ["F1", "F1Ign", "AUC", "AUCIgn"]
	row = [res_dict_dev[colname] for colname in columns]
	rows = [row]
	print(tabulate(rows, headers=columns, tablefmt="rst", floatfmt=".4f"))

	theta = res_dict_dev['opt_theta']
	print("Threshold", theta)
	#f1, precision, recall = con.test(model, input_theta=theta)
	#print(f"F1: {f1}\tPrecision: {precision}\tRecall:{recall}")

	print("Test ==============================================")
	con.load_test_data(split_dev=True, dev=False)
	res_dict = con.full_test(model, input_theta= theta)


	row = [res_dict[colname] for colname in columns]
	rows = [row]
	print(tabulate(rows, headers=columns, tablefmt="rst", floatfmt=".4f"))

	con.publish(model, f"{save_name}", theta)






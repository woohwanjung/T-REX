import config
import models
import argparse


def get_ext_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type = str, default = 'T-REX', help = 'name of the model (T-REX|BiLSTM|LSTM|ContextAware|CNN)')
	parser.add_argument('--save_name', type = str)

	parser.add_argument('--train_prefix', type = str, default = 'dev_train')
	parser.add_argument('--test_prefix', type = str, default = 'dev_dev')

	parser.add_argument("--batch_size", type=int, default=12)
	parser.add_argument("--hidden_size", type=int, default=128)
	parser.add_argument("--learning_rate", type=float, default=1e-5)
	parser.add_argument("--prediction_opt", type=int, default= models.TREX.PREDICTION_OPT_PREDICT_SMOOTHMAX)

	return parser

model_dict = {
	'CNN': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
	'BERT':models.Bert_Ext,
	models.TREX.name: models.TREX
}


if __name__ == "__main__":
	parser = get_ext_parser()
	args = parser.parse_args()
	print(args)

	if args.model_name == models.TREX.name:
		con = config.TopicAwareConfig(args)
	else:
		con = config.PairwiseConfig(args)

	if args.train_prefix == "dev_train":
		con.set_max_epoch(100)
		con.set_test_epoch(5)
	elif args.train_prefix == "train":
		con.set_max_epoch(10)
		con.set_test_epoch(1)

	con.load_train_data()
	con.load_test_data()

	con.train(model_dict[args.model_name])

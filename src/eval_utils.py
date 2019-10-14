import numpy as np

def evaluate(pred, truth):
	total = pred.shape[0]
	res1 = pred + truth
	res2 = pred - truth
	assert res1.shape == pred.shape
	true_pos = res1[res1 == 2].shape[0]
	true_neg = res1[res1 == 0].shape[0]
	false_pos = res2[res2 == 1].shape[0]
	false_neg = res2[res2 == -1].shape[0]

	precision = true_pos/(true_pos + false_pos)
	recall = true_pos/(true_pos + false_neg)
	acc = (true_pos + true_neg)/total
	f1 = 2*(precision*recall)/(precision + recall)
	print("Precision: {}\t\t Recall: {}\t\t F1 Score: {}\t\t Accuracy: {}".format(precision, recall, f1, acc))

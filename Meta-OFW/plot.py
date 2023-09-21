import copy
import pickle

import numpy as np

from headers import *

FLAGS = flags.FLAGS

class Plot:
	def __init__(self, xlabel, par=None):
		self.xlabel = xlabel
		self.par = par
		self.num_point = 200 # the maximum number of points in the picture
		self.start = 0

	def plot_loss(self, mean_stds, label_list, y_label):
		plt.style.use('seaborn-colorblind')
		plt.grid()
		self.unit = int(len(mean_stds[0][0]) / self.num_point)
		mean_list, std_list = [], []
		for mean, std in mean_stds:
			mean_tmp, std_tmp = copy.deepcopy(mean), copy.deepcopy(std)
			mean_tmp = np.asarray([mean[int(len(mean)/self.num_point * i)] for i in range(self.num_point)])
			std_tmp = np.asarray([std[int(len(std) / self.num_point * i)] for i in range(self.num_point)])
			mean_list.append(mean_tmp)
			std_list.append(std_tmp)
			# print(std_tmp)
		if FLAGS.SAVE_PLOT_DATA:
			if FLAGS.ENV == 'lds':
				self.filename = FLAGS.RESULT_BASE + FLAGS.ENV + '/{}-{}-{}'.format(FLAGS.NOISE, y_label, FLAGS.COST)
				# self.filename = FLAGS.RESULT_BASE + FLAGS.ENV + '/{}-{}-{}-{}-{}'.format(y_label, FLAGS.COST, self.par["seed"], self.par["step_pool_scale"], self.par["lr_meta_scale"])
			elif FLAGS.ENV in ['pendulum']:
				self.filename = FLAGS.RESULT_BASE + FLAGS.ENV + '/{}-{}-{}-{}'.format(y_label, self.par["seed"], self.par["step_pool_scale"], self.par["lr_meta_scale"])
			with open(self.filename, 'wb') as f:
				pickle.dump((mean_list, std_list, label_list), f)
		plt.xlabel(self.xlabel)
		plt.ylabel(y_label)
		xaxis = np.arange(0, len(mean_list[0]))
		for i in range(len(mean_list)):
			plt.plot(self.start + self.unit * xaxis, mean_list[i], label=label_list[i])
		for i in range(len(mean_list)):
			plt.fill_between(self.start + self.unit * xaxis, mean_list[i] - std_list[i], mean_list[i] + std_list[i],
							 alpha=0.15)
		plt.legend(loc='upper left')
		plt.savefig(self.filename + '.pdf')
		plt.show()

	def load_and_plot(self, filename):
		file = open(filename, 'rb')
		(mean_list, std_list, label_list) = pickle.load(file)
		self.__plot(mean_list, std_list, label_list)

def plot_time():
	a = [[519.8808705806732, 4.3695942950559665], [519.3213590145111, 2.3375850750776848],
		 [4951.631391620636, 14.473066183034424]]
	b = [[471.9238088130951, 5.538362921447853], [468.0896071910858, 10.893642780644408],
		 [5792.500298690796, 97.86369369599541]]
	c = [[212.95196380615235, 4.527414683196139], [222.21189889907836, 3.745830757037295],
		 [2151.375246191025, 92.48311067860402]]
	times = [a, b, c]
	mean1, std1 = np.around([x[0][0] for x in times]), [x[0][1] for x in times]
	mean2, std2 = np.around([x[1][0] for x in times]), [x[1][1] for x in times]
	mean3, std3 = np.around([x[2][0] for x in times]), [x[2][1] for x in times]

	dataset_labels = ['LDS, Slow', 'LDS, Abrupt', 'Pendulum']
	method_labels = ['OGD', 'Scream', 'Scream-N']
	plt.style.use('seaborn-colorblind')
	err_par = dict(elinewidth=0.5, ecolor='black', capsize=3)
	colors = ['#0B74B3', '#D55E00', '#FFBC00']
	width, linewidth, edgecolor = 0.25, 1, 'black'
	mpl.rcParams['hatch.linewidth'] = 0.5

	x = np.arange(len(dataset_labels))
	bar1 = plt.bar(x - width, mean1, width, label=method_labels[0], yerr=std1, error_kw=err_par, color=colors[0],
				   linewidth=linewidth, edgecolor=edgecolor)
	bar2 = plt.bar(x, mean2, width, label=method_labels[1], yerr=std2, error_kw=err_par, color=colors[1],
				   linewidth=linewidth,
				   edgecolor=edgecolor, hatch='///')
	bar3 = plt.bar(x + width, mean3, width, label=method_labels[2], yerr=std3, error_kw=err_par, color=colors[2],
				   linewidth=linewidth, edgecolor=edgecolor, hatch='\\\\\\')
	for b in bar1:
		height = b.get_height()
		plt.annotate('{}'.format(height), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, 3),
					 textcoords="offset points", color='black', ha='center', va='bottom')
	for b in bar2:
		height = b.get_height()
		plt.annotate('{}'.format(height), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, 3),
					 textcoords="offset points", color='black', ha='center', va='bottom')
	for b in bar3:
		height = b.get_height()
		plt.annotate('{}'.format(height), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, 4),
					 textcoords="offset points", color='black', ha='center', va='bottom')

	plt.ylabel('Time (seconds)')
	plt.xticks(x, labels=dataset_labels)
	plt.legend()
	plt.savefig('./results/time.pdf')
	plt.show()

if __name__ == '__main__':
    plot_time()
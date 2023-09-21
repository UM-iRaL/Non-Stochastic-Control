from headers import *

num_point = 200
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"

def __plot(start, T, mean_list, std_list, label_list, title):
    unit = int((T - start) / num_point)
    plt.grid(linestyle=':', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Loss')

    xaxis = np.arange(0, len(mean_list[0]))
    for i in range(len(mean_list)):
        plt.plot(start + unit * xaxis, mean_list[i], label=label_list[i])
    for i in range(len(mean_list)):
        plt.fill_between(start + unit * xaxis, mean_list[i] - std_list[i], mean_list[i] + std_list[i], alpha=0.15)
    plt.legend(loc='upper left')

def plot():
    label_list = ['OGD', 'Ader', 'Scream.light', 'Scream.vanilla']
    T = 10000

    plt.figure()

    result_base = './results/lds/'
    fn = 'CumulativeLoss-abrupt-0-100-1'
    f = open(result_base + fn, 'rb')
    (mean_list, std_list, _) = pickle.load(f)
    __plot(0, T, mean_list, std_list, label_list, '')

    plt.savefig(result_base + 'LDS (abrupt).pdf')
    plt.show()

if __name__ == '__main__':
    plot()
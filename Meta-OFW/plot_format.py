import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"

plt.grid(linestyle=':', linewidth=0.5)
plt.xlabel('Iteration')
plt.ylabel('Cumulative Loss')

plt.fill_between(alpha=0.15) # alpha = 0.15
plt.legend(loc='upper left')
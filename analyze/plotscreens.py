import pickle 
from expyfun import analyze as ea
import matplotlib.pyplot as plt

for i in range(3):
    temp = open((str(i) + 'screenshot.obj'), 'r')
    object = pickle.load(temp)
    plt.ion()
    ea.plot_screen(object)
    plt.savefig(str(i) + 'screenshot.pdf')


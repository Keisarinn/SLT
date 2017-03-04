import numpy as np
import matplotlib.pylab as plt

import matplotlib 
matplotlib.rcParams.update({'font.size': 22})

plot_folder = '../plots/'

samples = [100,500,1000,2000,3000]
times = [[0.1668,0.1775],[0.62266,0.63025],[2.0756,1.96481],[8.6981,8.45723],[20.7987,20.62265]]
avg_times = np.mean(times,axis=1)

n2 = [((x/1000)**2)*avg_times[2] for x in samples]
n3 = [((x/1000)**3)*avg_times[2] for x in samples]

fig = plt.figure(figsize=(10,7))
plt.plot(samples,avg_times,label='LLE time')
plt.plot(samples,n2,label='O(n'+u"\u00B2"+')')
plt.plot(samples,n3,label='O(n'+u"\u00B3"+')')
plt.legend(loc=2)
plt.title('LLE performance')
plt.xlabel('Samples')
plt.ylabel('Seconds')
fig.savefig(plot_folder+"LLEperformance.png")
plt.show()
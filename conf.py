from random import randint
import numpy as np
import pylab as pl
import scipy.stats as stats
import sys

fancy = False
TRIALS = 500
if len(sys.argv)>1:
    TRIALS = int(sys.argv[1])

if len(sys.argv)>2 and sys.argv[2]=='-fancy':
    fancy=True

prices = []

f = open("prices.txt")

for line in f:
    v = float(line.strip())
    prices.append(v)

def sample(data):
    samp = []
    for x in range(0,len(data)-1):
        samp.append(data[randint(0,len(data)-1)])
    return samp

def trials(num):
    nums = []
    for x in range(0,num):
        means = avg(sample(prices))
        nums.append(means)
    return nums

def avg(set):
    avg = sum(set) / len(set)
    return avg

def normpdf(x,mean,sd):
    out = (1/(sd*(np.sqrt(2*np.pi))))*(np.e**((-((x-mean)**2))/float((2*(sd**2)))))
    return out

print normpdf(1,0,1)

print np.pi

sorted = sorted(trials(TRIALS))

low = int(len(sorted)*.025)
high = int(len(sorted)*.975)

inside = []
for e in range(low,high):
    inside.append(sorted[e])


print(inside[0], inside[-1])

mean = avg(sorted)
sd = np.std(sorted)
l = inside[0]
r = inside[-1]

fig = pl.figure()
ax = fig.add_subplot(111)

x = np.arange(1.05, 1.25, .001 )
pl.axis([1.10,1.201,0,30])
x = np.arange(1.05, 1.25, .001 )
y = stats.norm.pdf(x,mean,sd)
#y = normpdf(x,mean,sd)


pl.plot(x,y,color='red')

z = [l, r]
s = [0,0]
pl.plot(z,s, "D")

left = (mean - (1.96*sd))
right = (mean + (1.96*sd))

ci_x = np.arange(left, right, .001)
ci_y = normpdf(ci_x, mean, sd)
pl.fill_between(ci_x, ci_y, color = '#F8ECE0')

pl.text(.02,.95,'$TRIALS = %d$' % TRIALS, transform = ax.transAxes)
pl.text(.02, .9, '$mean(prices)$ = %f' % np.mean(prices), transform = ax.transAxes)
pl.text(.02, .85, '$mean(\\overline{X})$ = %f' % np.mean(sorted), transform = ax.transAxes)
pl.text(.02, .80, '$stddev(\\overline{X})$ = %f' % np.std(sorted, ddof=1), transform = ax.transAxes)
pl.text(.02,.75,'95%% CI = $%1.2f \\pm 1.96*%1.3f$' % (np.mean(sorted), np.std(sorted,ddof=1)), transform = ax.transAxes)
pl.text(.02,.70,'95%% CI = $%1.2f \\ %1.2f$' % (np.mean(sorted)-1.96*np.std(sorted), np.mean(sorted) + 1.96*np.std(sorted)), transform = ax.transAxes)


pl.text(1.135, 11.5, "Expected", fontsize=16)
pl.text(1.135, 10, "95% CI $\\mu \\pm 1.96\\sigma$", fontsize=16)
pl.title("95% Confidence Intervals: $\\mu \\pm 1.96\\sigma$", fontsize=16)

ax.annotate("Empirical 95% CI",xy=(inside[0], .3),xycoords="data",xytext=(1.13, 4), textcoords='data', arrowprops = dict(arrowstyle="->",connectionstyle="arc3"),fontsize = 16)

pl.savefig('conf-'+str(TRIALS)+('basic' if not fancy else'')+'.pdf', format='pdf')

pl.show()




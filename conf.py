from random import randint
import numpy as np
import pylab as plt
import sys
import scipy.stats as stats

fancy = False
TRIALS = 50
if len(sys.argv) > 1:
    TRIALS = int(sys.argv[1])

if len(sys.argv) > 2 and sys.argv[2] == '-fancy':
    fancy = True

prices = []
f = open("prices.txt")
for line in f:
    v = float(line.strip())
    prices.append(v)


def sample(data):
    samp = []
    for x in range(0, len(data)):
        samp.append(data[randint(0, len(data)-1)])
    return samp

def trials(num):
    nums = []
    for x in range(0, num):
        means = avg(sample(prices))
        nums.append(means)
    return nums


def avg(set):
    avg = sum(set) / len(set)
    return avg


def normpdf(x, mean, sd):
    out = (1 / float((sd * (np.sqrt(2 * np.pi))))) * (np.e ** ((-((x - mean) ** 2)) / float((2 * (sd ** 2)))))
    return out


X_ = sorted(trials(TRIALS))
low = int(TRIALS * .025)
high = int(TRIALS * .975)-1

inside = []
for e in range(low, high):
    inside.append(X_[e])

l = inside[0]
r = inside[-1]

print(l, r)

mean = avg(X_)
sd = np.std(X_)


fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(1.05, 1.25, .001)
plt.axis([1.10, 1.201, 0, 30])
x = np.arange(1.05, 1.25, .001)
y = stats.norm.pdf(x,mean,sd)
plt.plot(x, y, color='red')

z = [l, r]
s = [0, 0]
plt.plot(z, s, "D")

left = (mean - (1.96 * sd))
right = (mean + (1.96 * sd))

ci_x = np.arange(left, right, .001)
ci_y = normpdf(ci_x, mean, sd)
plt.fill_between(ci_x, ci_y, color='#F8ECE0')



plt.text(.02,.95,'$TRIALS = %d$' % TRIALS, transform = ax.transAxes)
plt.text(.02,.9, '$mean(prices)$ = %f' % np.mean(prices), transform = ax.transAxes)
plt.text(.02,.85, '$mean(\\overline{X})$ = %f' % np.mean(X_), transform = ax.transAxes)
plt.text(.02,.80, '$stddev(\\overline{X})$ = %f' %np.std(X_,ddof=1), transform = ax.transAxes)
plt.text(.02,.75, '95%% CI = ($%1.2f,\\ %1.2f$)' % (left, right), transform = ax.transAxes)
plt.text(1.135, 11.5, "Expected", fontsize=16)
plt.text(1.135, 10, "95% CI $\\mu \\pm 1.96\\sigma$", fontsize=16)
plt.title("95% Confidence Intervals: $\\mu \\pm 1.96\\sigma$", fontsize=16)
ax.annotate("Empirical 95% CI", xy=(inside[0], .3), xycoords="data",xytext=(1.13,4), textcoords='data'
            ,arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), fontsize=16)
plt.savefig('bootstrap-' + str(TRIALS) + ('basic' if not fancy else'') + '.pdf', format='pdf')
plt.show()

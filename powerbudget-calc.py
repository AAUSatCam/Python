from ast import For
import math
import matplotlib
from matplotlib import pyplot

pass_numbers = 3 # 3 passes each block
pass_duration = 10 # 10 min duration
pass_pause = 90 # 1,5 hour between passes
pass_short_orbit_pause = 6*60 # approx 6 hours after first block
pass_long_orbit_pause = 11*60 # approx 11 hours after second block

display_num_blocks = 3

power_consumption = 2.9 # Raspberry pi uses 2.9W during max load

x_axis_calc = 24*60 # 60 minute ticks, 24 hours

full_orbit = 2*pass_numbers*(pass_duration + pass_pause) + pass_long_orbit_pause + pass_short_orbit_pause

x_axis = list(range(0, x_axis_calc))

y_axis = []

y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)

y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)

y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)

y_axis.extend([0]*pass_short_orbit_pause)

y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)
y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)
y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)

y_axis.extend([0]*pass_long_orbit_pause)

y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)
y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)
y_axis.extend([0]*pass_pause)
y_axis.extend([power_consumption]*pass_duration)

y_axis.extend([0]*pass_short_orbit_pause)

avg_power = [sum(y_axis) / len(y_axis)]*len(y_axis)

print(full_orbit/60)

pyplot.plot(y_axis, '-', color='k')
pyplot.plot(avg_power, ':', color='k')
pyplot.xlim(0,1715)
pyplot.xlabel("Time [m]", fontsize=20)
pyplot.xticks(fontsize=20)
pyplot.ylabel("Power [W]", fontsize=20)
pyplot.yticks(fontsize=20)
pyplot.axvline(x=full_orbit, linestyle='--', color='k')


pyplot.annotate('peak: 2.9W', xy=(959, 2.9), xytext=(1100, 2.5),
             arrowprops=dict(arrowstyle="->", relpos=(0,0.5)), fontsize=20, 
             )

pyplot.annotate('avg: 0.115W', xy=(1200, 0.1145), xytext=(1200, 0.65),
             arrowprops=dict(arrowstyle="->", relpos=(0,0.5)), fontsize=20,
             )

pyplot.annotate('Start of new orbit cycle', xy=(full_orbit, 1.25), xytext=(1000, 1.5),
             arrowprops=dict(arrowstyle="->", relpos=(1,0)), fontsize=20,
             )

pyplot.show()






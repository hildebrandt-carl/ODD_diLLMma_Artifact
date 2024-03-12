import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

OpenPilot_comma_ai_number_human_inspection  = [0, 4, 7, 100]
OpenPilot_2k19_number_human_inspection      = [1, 1, 13, 100]
OpenPilot_utah_number_human_inspection      = [0, 8, 7, 100]
number_human_inspection = [OpenPilot_comma_ai_number_human_inspection, 
                           OpenPilot_2k19_number_human_inspection,
                           OpenPilot_utah_number_human_inspection]

OpenPilot_comma_ai_failures_found_in_odd    = [0, 0, 2, 19]
OpenPilot_2k19_failures_found_in_odd        = [0, 0, 8, 36]
OpenPilot_utah_failures_found_in_odd        = [0, 1, 3, 16]
failures_found_in_odd = [OpenPilot_comma_ai_failures_found_in_odd, 
                         OpenPilot_2k19_failures_found_in_odd,
                         OpenPilot_utah_failures_found_in_odd]

total_images = [100, 100, 100]
total_failures = [19, 36, 16]

colors = ['C0', 'C1', 'C2', 'C3']
shapes = ['o', 's', '^'] 


# Create a figure
plt.figure(figsize=(17, 12))

# Create broken axes
bax = brokenaxes(ylims=((-1, 31), (89, 101)), xlims=((-1, 31), (89, 101)), despine=False)

for i in range(3):
    for j in range(4):
        # Set the color and shape
        s = shapes[i]
        c = colors[j]

        bax.scatter((number_human_inspection[i][j]/total_images[i])*100, (failures_found_in_odd[i][j]/total_failures[i])*100, marker=s, color=c, s=1000)


x = np.arange(100)
bax.plot(x, x, linestyle="dashed", color='C3', linewidth=6)

# Create custom legends
shape_legend = [mlines.Line2D([0], [0], color='black', marker=shape, linestyle='None', markersize=35) for shape in shapes]
color_legend = [mpatches.Patch(color=color) for color in colors[:3]]

# Custom line for 'Human' with dashed red line
human_line = mlines.Line2D([], [], color='red', linestyle='dashed', linewidth=6)

# Add 'Human' line to the color legend
color_legend.append(human_line)

# Add legends to the plot
legend1 = plt.legend(shape_legend, ['comma.ai 2016', 'comma.ai 2k19', 'JUtah'], loc='upper left', fontsize=35)
plt.gca().add_artist(legend1)  # Add the first legend manually
plt.legend(color_legend, ['Vicuna', 'Llama 2', 'ChatGPT-4V', 'Human'], loc='lower right', fontsize=35)

# Customize the plot
bax.set_xlabel('Images Requiring Human Inspection (%)', fontsize=35, labelpad=50)
bax.set_ylabel('Failure-Inducing Inputs in ODD (%)', fontsize=35, labelpad=50)

bax.tick_params(axis='both', which='major', labelsize=30)

# Show the grid
bax.grid()

# Show the plot
plt.savefig(f"./results/rq1b_time.png")

plt.show()
plt.close()
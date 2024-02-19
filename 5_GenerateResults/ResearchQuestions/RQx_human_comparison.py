import matplotlib.pyplot as plt

# Data
human_inside = [19, 36, 16]
human_outside = [81, 64, 84]
dataset_names = ["comma.ai 2016", "comma.ai 2k19", "JUtah"]

# Plot configuration
fig = plt.figure(figsize=(17, 12))
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='1', color='gray')
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')

# Bar plot
bar_inside = plt.bar(dataset_names,
                     human_inside,
                     label='Inside ODD',
                     color='C3',
                     edgecolor='black',
                     linewidth=5)
plt.bar(dataset_names,
        human_outside,
        bottom=human_inside,
        label='Outside ODD',
        color='C0',
        edgecolor='black',
        linewidth=5)

# Adding labels on top of the bars
for rect in bar_inside:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., height,
             '%d' % int(height),
             ha='center', va='bottom', fontsize=50, color="white")

# Setting tick and label size
plt.xticks(dataset_names, fontsize=30)
plt.tick_params(axis='y', labelsize=30) 
plt.ylabel('Number of Failure-Inducing Inputs', fontsize=42)
plt.xlabel('Dataset', fontsize=42)

# Set the y-limit
plt.ylim([0, 100])

# Add legend
plt.legend(fontsize=35, loc="upper right")

# Display the plot
plt.tight_layout()
plt.savefig(f"./results/rq1a_human.png")
plt.show()
plt.close()

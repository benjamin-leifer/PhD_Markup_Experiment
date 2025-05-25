import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
import matplotlib.dates as mdates

# Define tasks and timeline with milestones and color coding by goal
tasks = [
    "Coin cell setup (-21°C)",
    "Cathode operando Raman setup",
    "-32°C cycling tests",
    "Electrolyte optimization tests",
    "-51°C cycling tests",
    "Anode material evaluations",
    "Additive research",
    "Zn-ion GPE manuscript",
    "Low Temp Li Ion manuscript"
]
start_dates = [
    "2024-01-01", "2024-01-08", "2024-01-15",
    "2024-01-22", "2024-02-01", "2024-02-08",
    "2024-02-15", "2024-01-10", "2024-02-05"
]
end_dates = [
    "2024-01-07", "2024-01-14", "2024-01-21",
    "2024-01-31", "2024-02-07", "2024-02-14",
    "2024-02-28", "2024-01-31", "2024-03-15"
]
milestones = [
    {"task": "Zn-ion GPE manuscript", "date": "2024-01-20", "label": "5 pages done"},
    {"task": "Low Temp Li Ion manuscript", "date": "2024-02-28", "label": "First draft done"}
]
colors = {
    "Low Temp Experiments": "red",
    "Proposal": "blue",
    "Manuscripts": "green"
}
task_colors = [
    "red", "red", "red", "red", "red", "red", "red",
    "green", "green"
]

# Convert dates to matplotlib format
start_dates = [date2num(datetime.strptime(date, "%Y-%m-%d")) for date in start_dates]
end_dates = [date2num(datetime.strptime(date, "%Y-%m-%d")) for date in end_dates]
for milestone in milestones:
    milestone["date"] = date2num(datetime.strptime(milestone["date"], "%Y-%m-%d"))

# Create Gantt chart
fig, ax = plt.subplots(figsize=(12, 8))
for i, task in enumerate(tasks):
    ax.barh(i, end_dates[i] - start_dates[i], left=start_dates[i], align='center', color=task_colors[i], edgecolor='black')

# Add milestone lines and labels within task boxes
for milestone in milestones:
    task_index = tasks.index(milestone["task"])
    ax.axvline(milestone["date"], ymin=task_index / len(tasks), ymax=(task_index + 1) / len(tasks), color='black', linestyle='--', linewidth=1.5)
    ax.text(milestone["date"], task_index, f"  {milestone['label']}", verticalalignment='center', fontsize=10, color='black')

# Format chart
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.xlabel('Timeline')
plt.ylabel('Tasks')
plt.title('Gantt Chart with Milestones and Color Coding by Goal')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
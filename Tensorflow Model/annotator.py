import json
import matplotlib.pyplot as plt
import numpy as np

# Open coordinates_data.txt in read mode
with open("coordinates_data.txt", "r") as f:
    # arrs is an empty list to store the lines from the file along with the annotations.
    arrs = []
    # For each line in the file
    for line in f:
        # Append the line to arrs after removing the newline character
        arrs.append(line.replace("\n", ''))
        # Parse the JSON string in the line to create a Python list,
        # then convert it to a NumPy array arr.
        arr = np.array(list(json.loads(line)))
        # Extract x and y coordinates from the array
        x = arr[:, 0]
        y = arr[:, 1]
        # Plot the first point with a red circle
        plt.plot(x[0], y[0], marker='o', color='red')
        # Plot the path of the hand using the x and y coordinates
        plt.plot(x, y)
        # Plot the last point with a green cross
        plt.plot(x[-1], y[-1], marker='x', color='green')
        # Set the y-axis limit to [0, 480] and the x-axis limit to
        # [0, 640] to match the frame dimensions
        plt.ylim([0, 480])
        plt.xlim([0, 640])
        # Display the plot
        plt.show()
        # Prompt the user to enter a gesture annotation,
        # which is appended to the current line in arrs.
        inp = input('Enter the gesture: ')
        arrs[-1] += f", {inp}\n"

# Saving Annotated Data
# Open coordinates_data.txt in write mode
with open("coordinates_data.txt", "w") as f:
    # Write each line in arrs back to the file
    # This includes the original data plus the gesture annotation
    for line in arrs:
        f.write(line)

import matplotlib.pyplot as plt
import os

# Show all pipeline graphs
for i in range(1, 13):
    image_path = f"pipeline{i}_graph.png"
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pipeline {i} - Flow Difference per Hour")
        plt.show()
    else:
        print(f"Graph for Pipeline {i} not found!")

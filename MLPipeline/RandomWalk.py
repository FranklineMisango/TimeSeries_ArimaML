# import the required libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")



class RandomWalk:

    def random_walk(self):
        # Random Walk
        output = "./Output/"
        walk = [99]

        noise1 = []
        for i in range(1900):
            # Create random noise
            noise = -1 if np.random.random() < 0.5 else 1
            noise1.append(noise)
            walk.append(walk[-1] + noise)
        plt.plot(walk)
        plt.figure(figsize=(20, 5))
        plt.savefig(output+"randomwalk.png")
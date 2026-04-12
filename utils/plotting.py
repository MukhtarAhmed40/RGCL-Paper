import matplotlib.pyplot as plt

def plot_accuracy(acc_list):
    plt.plot(acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("results/accuracy.png")

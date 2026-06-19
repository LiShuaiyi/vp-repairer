import matplotlib.pyplot as plt
import matplotlib

success_rate = [99.7, 95.19, 91.29]

if __name__ == "__main__":
    plt.barh(range(3), success_rate, height=0.7, color="gray", alpha=0.75)
    plt.yticks(range(3), ["R_G3", "R_G2", "R_G1"])
    plt.xlim(0, 100)
    plt.xlabel("Repairable rate")
    for x, y in enumerate(success_rate):
        plt.text(y + 0.2, x - 0.1, "%s" % y)
    plt.show()

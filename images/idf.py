import math
import matplotlib.pyplot as plt
import numpy as np


def get_idf(n_q, N):
    """Calculate IDF score given term frequency and total documents"""
    idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
    return idf


# Set up the parameters
N = 1000  # Total number of documents
n_q_values = np.arange(1, N + 1)  # Term occurs in 1 to N documents
idf_scores = [get_idf(n_q, N) for n_q in n_q_values]

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(n_q_values, idf_scores, "b-", linewidth=2)
plt.xlabel("Number of documents containing the term (n_q)")
plt.ylabel("IDF Score")
plt.title("Inverse Document Frequency (IDF) vs Term Frequency")
plt.grid(True, alpha=0.3)
plt.xlim(1, N)
plt.ylim(0, max(idf_scores) * 1.1)

# Add some annotations for key points
plt.annotate(
    f"Rare term (n_q=1): IDF={get_idf(1, N):.3f}",
    xy=(1, get_idf(1, N)),
    xytext=(200, get_idf(1, N) + 0.5),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
    color="red",
)

plt.annotate(
    f"Common term (n_q={N // 2}): IDF={get_idf(N // 2, N):.3f}",
    xy=(N // 2, get_idf(N // 2, N)),
    xytext=(N // 2 + 200, get_idf(N // 2, N) + 0.3),
    arrowprops=dict(arrowstyle="->", color="orange"),
    fontsize=10,
    color="orange",
)

plt.annotate(
    f"Very common term (n_q={N}): IDF={get_idf(N, N):.3f}",
    xy=(N, get_idf(N, N)),
    xytext=(N - 300, get_idf(N, N) + 0.2),
    arrowprops=dict(arrowstyle="->", color="green"),
    fontsize=10,
    color="green",
)

plt.tight_layout()
plt.savefig("idf.png")

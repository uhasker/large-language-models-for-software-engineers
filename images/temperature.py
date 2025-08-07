import matplotlib.pyplot as plt
import numpy as np
import math


def apply_temperature(logprobs, temperature):
    """Apply temperature scaling to log probabilities."""
    if temperature == 0:
        # Handle edge case: return one-hot distribution for highest logprob
        max_idx = np.argmax(logprobs)
        probs = np.zeros_like(logprobs)
        probs[max_idx] = 1.0
        return probs

    # Apply temperature scaling to logprobs
    scaled_logprobs = logprobs / temperature

    # Convert to probabilities using softmax with numerical stability
    max_logprob = np.max(scaled_logprobs)
    exp_logprobs = np.exp(scaled_logprobs - max_logprob)
    sum_exp = np.sum(exp_logprobs)
    return exp_logprobs / sum_exp


# Create a sample probability distribution (log probabilities)
# This represents a typical distribution from an LLM
n_tokens = 6
x = np.arange(n_tokens)

# Create a realistic log probability distribution
# Higher values (closer to 0) = higher probability
base_logprobs = np.array([-0.5, -0.8, -1.1, -1.5, -2.0, -2.8])

# Temperature values to visualize
temperatures = [0.1, 0.5, 1.0, 2.0]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot distributions for different temperatures
for i, temp in enumerate(temperatures):
    probs = apply_temperature(base_logprobs, temp)
    ax.plot(
        x,
        probs,
        linewidth=2.5,
        color=colors[i],
        label=f"T = {temp}",
        alpha=0.8,
        marker="o",
        markersize=6,
    )
    ax.fill_between(x, probs, alpha=0.2, color=colors[i])

# Customize the plot
ax.set_xlabel("Token Index", fontsize=14, fontweight="bold")
ax.set_ylabel("Probability", fontsize=14, fontweight="bold")
ax.set_title(
    "Effect of Temperature on Probability Distributions",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Add legend with explanation
legend = ax.legend(
    fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True
)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(0.9)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle="--")

# Set axis limits
ax.set_xlim(-0.2, n_tokens - 0.8)
ax.set_ylim(0, None)
ax.set_xticks(x)

# Add annotations explaining the effect
ax.annotate(
    "Lower temperature:\nMore deterministic\n(peaked distribution)",
    xy=(0, 0.8),
    xytext=(1.5, 0.7),
    arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=1.5),
    fontsize=11,
    ha="left",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

ax.annotate(
    "Higher temperature:\nMore random\n(flatter distribution)",
    xy=(4, 0.2),
    xytext=(2.5, 0.4),
    arrowprops=dict(arrowstyle="->", color="#96CEB4", lw=1.5),
    fontsize=11,
    ha="left",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig(
    "images/temperature.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

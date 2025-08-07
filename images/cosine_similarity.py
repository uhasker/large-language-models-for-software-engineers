import math
import matplotlib.pyplot as plt
import numpy as np


def get_norm(v):
    return math.sqrt(sum(x**2 for x in v))


def get_dot_product(v, w):
    return sum(v[i] * w[i] for i in range(len(v)))


def cosine_similarity(v, w):
    return get_dot_product(v, w) / (get_norm(v) * get_norm(w))


# Define the vectors
v = [1, 0]
w = [1, 1]

# Calculate cosine similarity
similarity = cosine_similarity(v, w)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot the vectors
ax.quiver(
    0,
    0,
    v[0],
    v[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="blue",
    width=0.005,
    label=f"v = {v}",
)
ax.quiver(
    0,
    0,
    w[0],
    w[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
    width=0.005,
    label=f"w = {w}",
)

# Add angle arc to show the angle between vectors
angle = math.acos(similarity)
theta = np.linspace(0, angle, 100)
arc_radius = 0.3
arc_x = arc_radius * np.cos(theta)
arc_y = arc_radius * np.sin(theta)
ax.plot(arc_x, arc_y, "gray", linewidth=2)

# Add angle label
mid_angle = angle / 2
label_x = (arc_radius + 0.1) * math.cos(mid_angle)
label_y = (arc_radius + 0.1) * math.sin(mid_angle)
ax.text(label_x, label_y, f"θ = {math.degrees(angle):.1f}°", fontsize=12, ha="center")

# Set equal aspect ratio and limits
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.2, 1.3)
ax.set_aspect("equal")

# Add grid and labels
ax.grid(True, alpha=0.3)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(f"Cosine Similarity Visualization", fontsize=14)

# Add legend
ax.legend(loc="upper right")

# Add text box with calculations
textstr = f"||v|| = {get_norm(v):.3f}\n||w|| = {get_norm(w):.3f}\nv · w = {get_dot_product(v, w)}\ncos(θ) = {similarity:.3f}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(
    0.02,
    0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=props,
)

plt.tight_layout()
plt.savefig("src/images/cosine_similarity.png", dpi=300, bbox_inches="tight")
plt.show()

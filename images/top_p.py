import drawsvg as draw

# Create the canvas
canvas_width = 1500
canvas_height = 1000
d = draw.Drawing(canvas_width, canvas_height, origin=(0, 0))

# Add white background
d.append(draw.Rectangle(0, 0, canvas_width, canvas_height, fill="#FFFFFF"))

# Create arrow marker
marker = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient="auto")
marker.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill="black", close=True))
d.append(marker)


# Helper to draw centered text in a box
def draw_box(
    x,
    y,
    width,
    height,
    text,
    box_fill="#ffffff",
    box_stroke="#000000",
    text_size=20,
    text_weight="normal",
):
    d.append(draw.Rectangle(x, y, width, height, fill=box_fill, stroke=box_stroke))
    d.append(
        draw.Text(
            text,
            text_size,
            x + width / 2,
            y + height / 2 + 7,
            center=True,
            valign="middle",
            font_weight=text_weight,
        )
    )


# Token data with probabilities
tokens = [
    {"token": "Apple", "prob": 0.5},
    {"token": "Banana", "prob": 0.3},
    {"token": "Cherry", "prob": 0.1},
    {"token": "Durian", "prob": 0.05},
    {"token": "Elderberry", "prob": 0.05},
]

# Calculate box dimensions and positions
# Canvas: 1500x1000
# 5 boxes in a row, centered vertically and horizontally
box_width = 200
box_height = 120
spacing = 50
total_width = 5 * box_width + 4 * spacing  # 1200 total width
start_x = (canvas_width - total_width) / 2  # 150
center_y = canvas_height / 2  # 500
box_y = center_y - box_height / 2  # 440

# Colors for included (top-p <= 0.9) and excluded tokens
included_color = "#DDE7FF"  # Light blue for included tokens
excluded_color = "#FFE6E6"  # Light red for excluded tokens

# Draw token boxes
cumulative_prob = 0
for i, token_data in enumerate(tokens):
    x = start_x + i * (box_width + spacing)
    cumulative_prob += token_data["prob"]

    # Determine if this token is included in top-p sampling (p=0.9)
    if cumulative_prob <= 0.9 or i <= 2:  # First 3 tokens are included
        box_color = included_color
    else:
        box_color = excluded_color

    # Create box text with token and probability
    box_text = f"{token_data['token']}\n(p={token_data['prob']})"

    draw_box(
        x, box_y, box_width, box_height, box_text, box_fill=box_color, text_size=16
    )

# Draw cutoff line between 3rd and 4th boxes
cutoff_x = start_x + 3 * (box_width + spacing) - spacing / 2
cutoff_top = box_y - 50
cutoff_bottom = box_y + box_height + 50

# Draw vertical dashed line
d.append(
    draw.Line(
        cutoff_x,
        cutoff_top,
        cutoff_x,
        cutoff_bottom,
        stroke="red",
        stroke_width=3,
        stroke_dasharray="10,5",
    )
)

# Add labels
d.append(
    draw.Text(
        "Included in top-p sampling (p=0.9)",
        18,
        start_x + (3 * box_width + 2 * spacing) / 2,
        box_y - 80,
        center=True,
        fill="#0066CC",
        font_weight="bold",
    )
)

d.append(
    draw.Text(
        "Excluded from sampling",
        18,
        start_x + 3 * (box_width + spacing) + (2 * box_width + spacing) / 2,
        box_y - 80,
        center=True,
        fill="#CC0000",
        font_weight="bold",
    )
)

# Add title
d.append(
    draw.Text(
        "Top-P Sampling with p=0.9",
        24,
        canvas_width / 2,
        100,
        center=True,
        font_weight="bold",
    )
)

# Add cumulative probability explanation
d.append(
    draw.Text(
        "Cumulative probability: 0.5 + 0.3 + 0.1 = 0.9 ≥ p",
        16,
        canvas_width / 2,
        box_y + box_height + 100,
        center=True,
        fill="#666666",
    )
)

# Save the drawing
d.save_svg("src/images/top_p.svg")

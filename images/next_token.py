import drawsvg as draw

# Canvas setup
canvas_width = 1500
canvas_height = 1000
d = draw.Drawing(canvas_width, canvas_height, origin=(0, 0))

# Background
d.append(draw.Rectangle(0, 0, canvas_width, canvas_height, fill="#FFFFFF"))

# Arrow marker
marker = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient="auto")
marker.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill="black", close=True))
d.append(marker)


# Helper to draw a labeled box
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


# === Box 1: Input Text ===
input_box = {
    "x": 450,
    "y": 750,
    "width": 600,
    "height": 100,
    "text": "How are you? I am ",
    "box_fill": "#F1F3F5",  # gray-200
}
draw_box(**input_box)

# === Box 2: Large Language Model ===
llm_box = {
    "x": 500,
    "y": 450,
    "width": 500,
    "height": 150,
    "text": "Large Language Model",
    "box_fill": "#DDE7FF",  # indigo-200
    "text_size": 28,
    "text_weight": "bold",
}
draw_box(**llm_box)

# === Arrows (drawn first so they appear behind probability badges) ===

arrow_margin = 10

# Input → LLM
d.append(
    draw.Lines(
        input_box["x"] + input_box["width"] / 2,
        input_box["y"],
        llm_box["x"] + llm_box["width"] / 2,
        llm_box["y"] + llm_box["height"] + arrow_margin,
        close=False,
        stroke="black",
        stroke_width=3,
        fill="none",
        marker_end=marker,
    )
)

# === Token Output Boxes ===
tokens = [
    ("fine", "0.7", 400),
    ("good", "0.2", 650),
    ("bad", "0.1", 900),
]
token_y = 100
token_width = 200
token_height = 100
token_fill = "#FFF6EE"  # sand-200

token_boxes = []
for i, (token, prob, x) in enumerate(tokens):
    # Draw the token box
    draw_box(
        x,
        token_y,
        token_width,
        token_height,
        token,
        box_fill=token_fill,
    )

    token_boxes.append(
        {"x": x, "y": token_y, "width": token_width, "height": token_height}
    )

# LLM → each token (arrows drawn before probability badges)
for box in token_boxes:
    d.append(
        draw.Lines(
            llm_box["x"] + llm_box["width"] / 2,
            llm_box["y"],
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] + arrow_margin,
            close=False,
            stroke="black",
            stroke_width=3,
            fill="none",
            marker_end=marker,
        )
    )

# === Probability badges (drawn last so they appear on top) ===
for i, (token, prob, x) in enumerate(tokens):
    # Draw probability badge below the token box
    prob_y = token_y + token_height + 30
    prob_width = 100
    prob_height = 40

    # Adjust x position based on which token this is
    if i == 0:  # "fine" - move right
        prob_x = x + (token_width - prob_width) / 2 + 25
    elif i == 2:  # "bad" - move left
        prob_x = x + (token_width - prob_width) / 2 - 25
    else:  # "good" - keep centered
        prob_x = x + (token_width - prob_width) / 2

    # Probability badge with rounded appearance
    d.append(
        draw.Rectangle(
            prob_x,
            prob_y,
            prob_width,
            prob_height,
            fill="#3E63DD",
            stroke="none",
            rx=20,
            ry=20,
        )
    )
    d.append(
        draw.Text(
            f"p = {prob}",
            16,
            prob_x + prob_width / 2,
            prob_y + prob_height / 2 + 5,
            center=True,
            valign="middle",
            font_weight="bold",
            fill="white",
        )
    )

# === Render and Save ===
d.set_render_size(canvas_width, canvas_height)
d.save_svg("next_token.svg")

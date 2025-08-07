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


# Input sentence box
input_box = {
    "x": 450,
    "y": 717,
    "width": 600,
    "height": 100,
    "text": "Hello, world",
    "box_fill": "#F1F3F5",  # gray-200 for input
}
draw_box(**input_box)

# Tokenizer box (bigger and more prominent)
tokenizer_box = {
    "x": 550,
    "y": 350,
    "width": 400,
    "height": 150,
    "text": "Tokenizer",
    "box_fill": "#DDE7FF",  # indigo-200 for tokenizer
    "text_size": 32,
    "text_weight": "bold",
}
draw_box(**tokenizer_box)

# Top token boxes
token_labels = ["'Hello'", "','", "' '", "'wo'", "'rl'", "'d'"]
box_width = 216
box_height = 100
gap = 20
margin_left = 52
top_y = 116

token_boxes = []
for i, label in enumerate(token_labels):
    x = margin_left + i * (box_width + gap)
    draw_box(
        x, top_y, box_width, box_height, label, box_fill="#FFF6EE"
    )  # sand-200 for tokens
    token_boxes.append({"x": x, "y": top_y, "width": box_width, "height": box_height})

# Arrow margin to prevent arrow tips from overlapping boxes
arrow_margin = 10

# Arrow from input to tokenizer (upward)
d.append(
    draw.Lines(
        input_box["x"] + input_box["width"] / 2,
        input_box["y"],
        tokenizer_box["x"] + tokenizer_box["width"] / 2,
        tokenizer_box["y"] + tokenizer_box["height"] + arrow_margin,
        close=False,
        stroke="black",
        stroke_width=3,
        fill="none",
        marker_end=marker,
    )
)

# Arrows from tokenizer to tokens (pointing downward to top of token boxes)
for box in token_boxes:
    d.append(
        draw.Lines(
            tokenizer_box["x"] + tokenizer_box["width"] / 2,
            tokenizer_box["y"],
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] + arrow_margin,
            close=False,
            stroke="black",
            stroke_width=3,
            fill="none",
            marker_end=marker,
        )
    )

# Display or save the drawing
d.set_render_size(canvas_width, canvas_height)
d.save_svg("tokens.svg")

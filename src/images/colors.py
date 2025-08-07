import drawsvg as draw

# Create the canvas
canvas_width = 1200
canvas_height = 2400
d = draw.Drawing(canvas_width, canvas_height, origin=(0, 0))

# Add white background
d.append(draw.Rectangle(0, 0, canvas_width, canvas_height, fill="#FFFFFF"))

# Color palettes
indigo_palette = [
    ("#EEF4FF", "indigo-100", "very light call-out background, inline code bg"),
    ("#DDE7FF", "indigo-200", "hover bg, subtle HR line"),
    ("#BBCEFF", "indigo-300", "charts fills, badges"),
    ("#99B4FF", "indigo-400", "secondary buttons, links on dark surfaces"),
    ("#3E63DD", "indigo-500", "headings, CTAs, pull-quotes"),
    ("#2C4DBA", "indigo-600", "active CTA / link hover"),
    ("#1B3796", "indigo-700", "focus ring, borders on light cards"),
    ("#102571", "indigo-800", "dark-mode body text"),
    ("#08154D", "indigo-900", "dark-mode background blocks"),
]

sand_palette = [
    ("#FFFBF6", "sand-100", "article intro, footnotes bg"),
    ("#FFF6EE", "sand-200", "call-out bg, quote block bg"),
    ("#FEECD9", "sand-300", "chart fills, soft tags"),
    ("#F2D5B2", "sand-400", "secondary button bg, blockquote border"),
    ("#DDB88E", "sand-500", "icons on light bg"),
    ("#C29D77", "sand-600", "hover state on sand-400"),
    ("#A58262", "sand-700", "dark-mode sand text"),
    ("#84674E", "sand-800", "dark-mode sand bg"),
    ("#5D4634", "sand-900", "dark-mode sand border"),
]

gray_palette = [
    ("#FCFCFD", "gray-50", "card interiors"),
    ("#F8F9FA", "gray-100", "article BG alt rows"),
    ("#F1F3F5", "gray-200", "hairlines, subtle dividers"),
    ("#E9ECEF", "gray-300", "code block borders"),
    ("#DEE2E6", "gray-400", "caption text"),
    ("#CED4DA", "gray-500", "UI chrome borders"),
    ("#ADB5BD", "gray-600", "secondary text"),
    ("#868E96", "gray-700", "muted headings"),
    ("#495057", "gray-800", "body text on tinted BGs"),
    ("#343A40", "gray-900", "dark-mode body text"),
]


# Helper function to draw a color swatch with label
def draw_color_swatch(x, y, color, token, description, width=300, height=60):
    # Color rectangle
    d.append(
        draw.Rectangle(
            x, y, width, height, fill=color, stroke="#000000", stroke_width=1
        )
    )

    # Token label (bold)
    d.append(draw.Text(token, 16, x + 10, y + 20, font_weight="bold", fill="#000000"))

    # Hex color
    d.append(draw.Text(color, 14, x + 10, y + 35, fill="#333333"))

    # Description (smaller text, right side)
    d.append(draw.Text(description, 12, x + width + 20, y + 30, fill="#666666"))


# Draw title
d.append(draw.Text("Color Palette", 24, 50, 50, font_weight="bold", fill="#000000"))

# Draw Indigo palette
y_start = 100
d.append(
    draw.Text(
        '1. Primary accent — "Soft Indigo"',
        20,
        50,
        y_start,
        font_weight="bold",
        fill="#3E63DD",
    )
)
for i, (color, token, description) in enumerate(indigo_palette):
    draw_color_swatch(50, y_start + 40 + i * 70, color, token, description)

# Draw Sand palette
y_start = 100 + 40 + len(indigo_palette) * 70 + 50
d.append(
    draw.Text(
        '2. Secondary accent — "Muted Sand"',
        20,
        50,
        y_start,
        font_weight="bold",
        fill="#DDB88E",
    )
)
for i, (color, token, description) in enumerate(sand_palette):
    draw_color_swatch(50, y_start + 40 + i * 70, color, token, description)

# Draw Gray palette
y_start = 100 + 40 + len(indigo_palette) * 70 + 50 + 40 + len(sand_palette) * 70 + 50
d.append(
    draw.Text(
        "3. Neutral ramp (cool gray)",
        20,
        50,
        y_start,
        font_weight="bold",
        fill="#868E96",
    )
)
for i, (color, token, description) in enumerate(gray_palette):
    draw_color_swatch(50, y_start + 40 + i * 70, color, token, description)

# Save the drawing
d.set_render_size(canvas_width, canvas_height)
d.save_svg("colors.svg")
print("Color palette visualization saved as 'colors.svg'")

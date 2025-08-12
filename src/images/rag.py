# RAG diagram (1500x1000) using drawsvg with correctly aligned arrowheads

import drawsvg as draw

W, H = 1500, 1000
d = draw.Drawing(W, H, origin=(0, 0))

# Add background with subtle gradient feel
d.append(draw.Rectangle(0, 0, W, H, fill="#FCFCFD"))

# ---------- Arrow marker (tip centered on line end) ----------
# Create marker with proper dimensions and positioning
arrow = draw.Marker(-2, -2, 4, 4, scale=1, orient="auto")
triangle = draw.Path(d="M -2,-2 L 2,0 L -2,2 Z", fill="#3E63DD", stroke="none")
arrow.append(triangle)
d.append(arrow)

# ---------- Geometry ----------
# Boxes
r_x, r_y, r_w, r_h = 350, 450, 220, 100  # Retriever
g_x, g_y, g_w, g_h = 820, 450, 240, 100  # Generator
d_x, d_y, d_w, d_h = 330, 650, 260, 100  # Document Store

# Centers/edges
r_cx, r_cy = r_x + r_w / 2, r_y + r_h / 2  # (460, 500)
g_cx, g_cy = g_x + g_w / 2, g_y + g_h / 2  # (940, 500)
y_flow = r_cy  # 500

# Text positions
prompt_xy = (60, 510)
response_xy = (1310, 510)

# Arrow coordinates
p_to_r = ((190, y_flow), (r_x, y_flow))
r_to_g = ((r_x + r_w, y_flow), (g_x, y_flow))
g_to_rsp = ((g_x + g_w, y_flow), (1300, y_flow))

down_q = ((r_cx - 25, r_y + r_h), (r_cx - 25, d_y))
up_res = ((r_cx + 25, d_y), (r_cx + 25, r_y + r_h))

curve_start = (190, y_flow)
curve_ctrl = ((curve_start[0] + g_x) / 2, 140)  # (505, 140)
curve_end = (g_x, y_flow)

# ---------- Draw boxes with modern styling ----------
# Retriever box - primary accent color
d.append(
    draw.Rectangle(
        r_x,
        r_y,
        r_w,
        r_h,
        rx=18,
        ry=18,
        fill="#EEF4FF",
        stroke="#3E63DD",
        stroke_width=3,
    )
)

# Generator box - secondary accent color
d.append(
    draw.Rectangle(
        g_x,
        g_y,
        g_w,
        g_h,
        rx=18,
        ry=18,
        fill="#FFF6EE",
        stroke="#DDB88E",
        stroke_width=3,
    )
)

# Document Store box - neutral with subtle tint
d.append(
    draw.Rectangle(
        d_x,
        d_y,
        d_w,
        d_h,
        rx=18,
        ry=18,
        fill="#F8F9FA",
        stroke="#868E96",
        stroke_width=3,
    )
)

# Labels with better typography and colors
d.append(
    draw.Text(
        "Retriever", 32, r_cx, r_cy + 10, center=True, fill="#3E63DD", font_weight="600"
    )
)
d.append(
    draw.Text(
        "Generator", 32, g_cx, g_cy + 10, center=True, fill="#DDB88E", font_weight="600"
    )
)
d.append(
    draw.Text(
        "Document",
        28,
        d_x + d_w / 2,
        d_y + 40,
        center=True,
        fill="#495057",
        font_weight="500",
    )
)
d.append(
    draw.Text(
        "Store",
        28,
        d_x + d_w / 2,
        d_y + 78,
        center=True,
        fill="#495057",
        font_weight="500",
    )
)
d.append(draw.Text("Prompt", 36, *prompt_xy, fill="#343A40", font_weight="600"))
d.append(draw.Text("Response", 36, *response_xy, fill="#343A40", font_weight="600"))


# ---------- Draw arrows with enhanced styling ----------
def line_arrow(p1, p2, color="#3E63DD", width=4):
    x1, y1 = p1
    x2, y2 = p2
    d.append(
        draw.Line(
            x1,
            y1,
            x2,
            y2,
            stroke=color,
            stroke_width=width,
            marker_end=arrow,
            stroke_linecap="round",
        )
    )


# Straight arrows with primary color
line_arrow(*p_to_r)
line_arrow(*r_to_g)
line_arrow(*g_to_rsp)

# Vertical bidirectional arrows with secondary color
line_arrow(*down_q, color="#868E96", width=3)
line_arrow(*up_res, color="#868E96", width=3)

# Curved top arrow (quadratic Bézier): Prompt → Generator with gradient feel
curve = draw.Path(
    stroke="#2C4DBA",
    fill="none",
    stroke_width=4,
    marker_end=arrow,
    stroke_linecap="round",
)
curve.M(*curve_start).Q(*curve_ctrl, *curve_end)
d.append(curve)

# Save
d.save_svg("rag.svg")
print("Saved to rag.svg")

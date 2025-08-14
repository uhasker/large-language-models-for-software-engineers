# SEO-optimized title for the article
TITLE = "Sliding-Window Chunking for Text: How to Keep Context Between Chunks"

# Draw an illustration of OVERLAP CHUNKING using drawsvg
# Canvas: 1500 x 1000

# ---------------------------
# Planned coordinates (in px)
# ---------------------------
# Canvas: (width=1500, height=1000)
# Document bar:
#   doc_x = 100, doc_y = 180, doc_w = 1300, doc_h = 60
#
# Chunk geometry (projected along the same horizontal axis as the document):
#   chunk_w = 520
#   overlap = 160
#   stride  = chunk_w - overlap = 360
# Chunk start x positions (aligned to document x):
#   starts = [doc_x + i*stride for i in range(3)] -> [100, 460, 820]
# Chunk end x positions:
#   ends   = [x + chunk_w for x in starts] -> [620, 980, 1340]
#
# Chunk rows (drawn BELOW the document for clarity):
#   chunk_h = 80
#   chunk_ys = [420, 560, 700]
#
# Overlap regions on the DOCUMENT (visual emphasis of shared context):
#   between chunk 1 & 2: x in [starts[1], ends[0]] -> [460, 620]
#   between chunk 2 & 3: x in [starts[2], ends[1]] -> [820, 980]
#
# Visual links (dashed guidelines) from chunk edges up to document edges:
#   vertical dashed lines at each start/end, from chunk_y to doc_y+doc_h
# ---------------------------

import drawsvg as draw

# --- Canvas ---
W, H = 1500, 1000
d = draw.Drawing(W, H, origin=(0, 0), displayInline=False)

# --- Background ---
d.append(draw.Rectangle(0, 0, W, H, fill="#f8fafc"))


# --- Helpers / Style ---
def txt(x, y, s, anchor="start", weight="normal"):
    d.append(
        draw.Text(
            s,
            22,
            x,
            y,
            center=anchor == "middle",
            right=anchor == "end",
            font_family="Inter, Arial, sans-serif",
            font_weight=weight,
        )
    )


def guide(x, y1, y2):
    d.append(
        draw.Line(
            x,
            y1,
            x,
            y2,
            stroke="#6b7280",
            stroke_width=2,
            stroke_dasharray="6,6",
            stroke_opacity=0.7,
        )
    )


def label_box(x, y, w, h, title, lines):
    d.append(
        draw.Rectangle(
            x, y, w, h, rx=12, ry=12, fill="#ffffff", stroke="#111827", stroke_width=2
        )
    )
    d.append(
        draw.Text(
            title,
            24,
            x + 16,
            y + 36,
            font_family="Inter, Arial, sans-serif",
            font_weight="600",
        )
    )
    yy = y + 68
    for line in lines:
        d.append(
            draw.Text(line, 20, x + 16, yy, font_family="Inter, Arial, sans-serif")
        )
        yy += 28


def overlap_band(x, y, w, h):
    # greenish band to show shared region
    d.append(
        draw.Rectangle(
            x,
            y,
            w,
            h,
            fill="#a7f3d0",
            fill_opacity=0.8,
            stroke="#10b981",
            stroke_width=2,
        )
    )


def chunk_rect(x, y, w, h, label, left_overlap_px=0, right_overlap_px=0):
    # base
    d.append(
        draw.Rectangle(
            x, y, w, h, rx=10, ry=10, fill="#e0f2fe", stroke="#1d4ed8", stroke_width=2
        )
    )
    # overlaps inside the chunk (left/right), shaded to highlight reused context
    if left_overlap_px > 0:
        d.append(
            draw.Rectangle(
                x,
                y,
                left_overlap_px,
                h,
                rx=10,
                ry=10,
                fill="#fde68a",
                fill_opacity=0.9,
                stroke="none",
            )
        )
    if right_overlap_px > 0:
        d.append(
            draw.Rectangle(
                x + w - right_overlap_px,
                y,
                right_overlap_px,
                h,
                rx=10,
                ry=10,
                fill="#fde68a",
                fill_opacity=0.9,
                stroke="none",
            )
        )
    # label
    d.append(
        draw.Text(
            label,
            20,
            x + w / 2,
            y + h / 2 + 6,
            font_family="Inter, Arial, sans-serif",
            center=True,
            font_weight="600",
            fill="#111827",
        )
    )


# --- Title ---
d.append(
    draw.Text(
        TITLE,
        34,
        W / 2,
        70,
        center=True,
        font_family="Inter, Arial, sans-serif",
        font_weight="700",
    )
)

# --- Document bar ---
doc_x, doc_y, doc_w, doc_h = 100, 180, 1300, 60
d.append(
    draw.Rectangle(
        doc_x,
        doc_y,
        doc_w,
        doc_h,
        rx=12,
        ry=12,
        fill="#f3f4f6",
        stroke="#111827",
        stroke_width=2,
    )
)
txt(doc_x + 12, doc_y + doc_h / 2 + 8, "Document", weight="600")

# --- Chunk geometry ---
chunk_w = 520
overlap = 160
stride = chunk_w - overlap
starts = [doc_x + i * stride for i in range(3)]
ends = [x + chunk_w for x in starts]
chunk_h = 80
chunk_ys = [420, 560, 700]

# --- Overlap bands on the document (shared context regions) ---
# Between chunk 1 & 2
overlap_band(starts[1], doc_y, overlap, doc_h)
# Between chunk 2 & 3
overlap_band(starts[2], doc_y, overlap, doc_h)

# --- Guidelines from chunk edges up to the document ---
for i in range(3):
    # left edge guideline
    guide(starts[i], chunk_ys[i], doc_y + doc_h)
    # right edge guideline
    guide(ends[i], chunk_ys[i], doc_y + doc_h)

# --- Chunks ---
# Chunk 1: right overlap only (with next)
chunk_rect(
    starts[0],
    chunk_ys[0],
    chunk_w,
    chunk_h,
    "Chunk 1",
    left_overlap_px=0,
    right_overlap_px=overlap,
)
# Chunk 2: left + right overlap (with prev and next)
chunk_rect(
    starts[1],
    chunk_ys[1],
    chunk_w,
    chunk_h,
    "Chunk 2",
    left_overlap_px=overlap,
    right_overlap_px=overlap,
)
# Chunk 3: left overlap only (with prev)
chunk_rect(
    starts[2],
    chunk_ys[2],
    chunk_w,
    chunk_h,
    "Chunk 3",
    left_overlap_px=overlap,
    right_overlap_px=0,
)

# --- Annotations ---
txt(
    100,
    360,
    "Each chunk slides forward by the stride while overlapping with the previous chunk to keep context.",
    weight="600",
)
txt(
    100,
    390,
    "Shaded yellow areas show reused text on chunk edges; green bands highlight the same overlaps on the document.",
)

# --- Save SVG ---
d.save_svg("simple_chunking.svg")

# If running in a Jupyter environment, also display inline:
# d

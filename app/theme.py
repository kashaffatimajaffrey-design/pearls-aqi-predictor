"""Visual theme for the dashboard.

Two blue progressions, both meaning something rather than decorating:

1. **Time of day.** The page background moves dawn -> day -> dusk -> night with
   the local clock, so the app feels like the sky it is describing.
2. **Forecast depth.** The three horizon cards deepen from light to dark blue as
   they reach further out, which also reads as rising uncertainty.

AQI values keep their EPA colours on top of this shell. Blue is the chrome; the
green/yellow/orange/red/purple/maroon scale is a public-health convention and
recolouring it would delete the danger signal that makes the page useful.
"""
from __future__ import annotations


def page_css(sky: dict) -> str:
    """Global stylesheet, parameterised by the current sky phase."""
    return f"""
<style>
  :root {{
    --sky-from: {sky['from']};
    --sky-to: {sky['to']};
    --ink: {sky['ink']};
    --panel: #ffffff;
    --panel-brd: rgba(15,39,64,.10);
    --deep: #0a1929;
  }}

  .stApp {{
    background: linear-gradient(165deg, var(--sky-from) 0%, var(--sky-to) 55%,
                                var(--sky-to) 100%);
    background-attachment: fixed;
  }}
  .block-container {{padding-top: 1.6rem; max-width: 1380px;}}

  /* EVERY piece of text drawn straight onto the gradient must follow the sky,
     or it disappears at night. Targeting only headings was not enough --
     st.metric, widget labels and slider ticks all default to dark ink and were
     invisible against the night background. Panel contents are re-darkened
     below, after these rules, so they still win on their own light surface. */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{color: var(--ink);}}
  .stApp > header {{background: transparent;}}

  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
  [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
    color: var(--ink) !important;
  }}
  [data-testid="stMetricLabel"] {{opacity: .8;}}

  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label {{color: var(--ink) !important;}}

  div[data-testid="stCaptionContainer"] p {{color: var(--ink); opacity: .78;}}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] strong {{color: var(--ink);}}

  /* Slider min/max ticks and the value bubble. */
  .stSlider [data-testid="stTickBar"], .stSlider [data-testid="stTickBar"] div,
  .stSlider div[data-baseweb="slider"] div {{color: var(--ink);}}

  .stRadio label p, .stMultiSelect label p, .stSelectbox label p,
  .stCheckbox label p, .stToggle label p {{color: var(--ink) !important;}}

  /* st.container(border=True) is the only reliable way to box Streamlit
     widgets, so it becomes the frosted panel. */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--panel);
    border: 1px solid var(--panel-brd) !important;
    border-radius: 16px;
    backdrop-filter: blur(9px);
    box-shadow: 0 2px 10px rgba(15,39,64,.07);
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] *,
  div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricLabel"],
  div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"],
  div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p {{
    color: var(--deep) !important;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] h1,
  div[data-testid="stVerticalBlockBorderWrapper"] h2,
  div[data-testid="stVerticalBlockBorderWrapper"] h3,
  div[data-testid="stVerticalBlockBorderWrapper"] h4 {{color: var(--deep);}}

  /* Frosted panel used for every content block. */
  .panel {{
    background: var(--panel);
    border: 1px solid var(--panel-brd);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(9px);
    box-shadow: 0 2px 10px rgba(15,39,64,.07);
    color: var(--deep);
  }}

  /* ---- hero ---- */
  .hero {{
    display: block;
    background: #ffffff;
    border: 1px solid var(--panel-brd);
    border-radius: 20px; padding: 1.3rem 1.6rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 3px 14px rgba(15,39,64,.09);
    color: var(--deep);
  }}
  .hero .num {{font-size: 3.6rem; font-weight: 800; line-height: .95;
               letter-spacing: -.02em;}}
  .hero .advice {{font-size:.83rem; opacity:.8; margin-top:.6rem;
                  padding-top:.6rem; border-top:1px solid rgba(10,25,41,.14);}}
  .hero .cat {{font-weight: 700; font-size: 1.02rem;}}
  .epa-dot {{display:inline-block; width:.72rem; height:.72rem; border-radius:50%;
             margin-right:.4rem; vertical-align:middle;
             box-shadow: 0 0 0 2px rgba(255,255,255,.9);}}

  /* ---- forecast cards: light -> dark with horizon ---- */
  .fc {{
    border-radius: 16px; padding: 1rem 1.1rem; color: #f2f8ff; height: 100%;
    box-shadow: 0 8px 20px rgba(10,25,41,.22);
    transition: transform .18s ease, box-shadow .18s ease;
    position: relative; overflow: hidden;
  }}
  .fc:hover {{transform: translateY(-4px); box-shadow: 0 14px 30px rgba(10,25,41,.3);}}
  .fc .day {{font-size:.72rem; text-transform:uppercase; letter-spacing:.09em;
             opacity:.85; font-weight:700;}}
  .fc .val {{font-size:2.5rem; font-weight:800; line-height:1.05; margin:.15rem 0;}}
  .fc .cat {{font-size:.82rem; font-weight:700;}}
  .fc .rng {{font-size:.7rem; opacity:.8; margin-top:.3rem;}}
  .fc .epa {{position:absolute; top:.85rem; right:.9rem; width:.85rem;
             height:.85rem; border-radius:50%;
             box-shadow:0 0 0 2px rgba(255,255,255,.75);}}

  /* ---- gamification ---- */
  .tile {{background: var(--panel); border:1px solid var(--panel-brd);
          border-radius:14px; padding:.85rem 1rem; text-align:center;
          color:var(--deep); height:100%;}}
  .tile .big {{font-size:1.9rem; font-weight:800; line-height:1;}}
  .tile .cap {{font-size:.7rem; text-transform:uppercase; letter-spacing:.07em;
               opacity:.7; font-weight:700; margin-top:.2rem;}}

  .badge {{
    display:flex; align-items:center; gap:.6rem;
    border-radius:12px; padding:.55rem .75rem; margin-bottom:.45rem;
    border:1px solid rgba(255,255,255,.5);
  }}
  .badge.on  {{background: linear-gradient(120deg,#1e88e5,#42a5f5); color:#fff;}}
  /* Unearned badges were translucent white over a dark sky, which muddied to
     grey-on-grey. Opaque enough to read, muted enough to still say "locked". */
  .badge.off {{background: rgba(255,255,255,.86); color:#3d4f5c;}}
  .badge.off .ico {{filter: grayscale(.85) opacity(.75);}}
  .badge .ico {{font-size:1.3rem; filter:saturate(1.1);}}
  .badge .ttl {{font-weight:700; font-size:.86rem;}}
  .badge .dsc {{font-size:.72rem; opacity:.85;}}

  .bar {{height:7px; border-radius:99px; background:rgba(10,25,41,.13);
         overflow:hidden; margin-top:.35rem;}}
  .bar > i {{display:block; height:100%; border-radius:99px;
             background:linear-gradient(90deg,#64b5f6,#1565c0);}}

  .grade {{font-size:2.6rem; font-weight:800; line-height:1;}}
  .pill {{display:inline-block; padding:.16rem .55rem; border-radius:99px;
          font-size:.68rem; font-weight:700; letter-spacing:.05em;
          text-transform:uppercase;}}
  .pill.live   {{background:#c8f7d4; color:#0b5f28;}}
  .pill.recent {{background:#fff3c4; color:#7a5b00;}}
  .pill.stale  {{background:#ffd6d6; color:#8b1a1a;}}

  .act {{display:flex; justify-content:space-between; align-items:center;
         padding:.4rem .1rem; border-bottom:1px dashed rgba(10,25,41,.13);
         font-size:.86rem;}}
  .act:last-child {{border-bottom:none;}}

  /* Readable sidebar against the gradient. */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1929, #143a5f);
  }}
  section[data-testid="stSidebar"] * {{color: #e3f2fd !important;}}

  /* Tab labels sit on the gradient and need an opaque chip plus an explicit
     colour -- inheriting leaves them invisible at night. Streamlit has moved
     this element from data-baseweb="tab" to data-testid="stTab" across
     versions, so both are targeted. */
  .stTabs [data-baseweb="tab-list"], div:has(> [data-testid="stTab"]) {{
    gap:.35rem; background: transparent;
  }}
  [data-testid="stTab"], .stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,.75); border-radius:10px 10px 0 0;
    padding:.45rem 1rem; font-weight:600;
  }}
  [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p {{
    color:#1c3d5a !important; font-weight:600; margin:0;
  }}
  [data-testid="stTab"][aria-selected="true"],
  .stTabs [aria-selected="true"] {{ background: rgba(255,255,255,.96); }}
  [data-testid="stTab"][aria-selected="true"] p,
  .stTabs [aria-selected="true"] p {{ color:#0d47a1 !important; font-weight:800; }}
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    background: transparent !important;
  }}

  /* The default header strip is opaque white and cuts the sky in half. */
  [data-testid="stHeader"] {{background: transparent;}}

  /* Inline code in the sidebar renders as a white block by default. */
  section[data-testid="stSidebar"] code {{
    background: rgba(255,255,255,.14) !important;
    color: #90caf9 !important; padding:.05rem .35rem; border-radius:5px;
  }}
</style>
"""


def hero_html(aqi: float, category: str, epa_color: str, advice: str,
              phase_label: str) -> str:
    return f"""
<div class="hero">
  <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;
              opacity:.65;font-weight:700">Current AQI &nbsp;&middot;&nbsp; {phase_label}</div>
  <div class="num">{aqi:.0f}</div>
  <div class="cat"><span class="epa-dot" style="background:{epa_color}"></span>{category}</div>
  <div class="advice">{advice}</div>
</div>
"""


def forecast_card(day_label: str, aqi: float, category: str, epa_color: str,
                  low: float, high: float, delta: float, blue: str) -> str:
    return f"""
<div class="fc" style="background:linear-gradient(150deg,{blue},{_darken(blue)})">
  <div class="epa" style="background:{epa_color}"></div>
  <div class="day">{day_label}</div>
  <div class="val">{aqi:.0f}</div>
  <div class="cat">{category}</div>
  <div class="rng">{delta:+.0f} vs now &nbsp;&middot;&nbsp; range {low:.0f}-{high:.0f}</div>
</div>
"""


def tile(value: str, caption: str, sub: str = "") -> str:
    extra = f"<div style='font-size:.7rem;opacity:.6;margin-top:.2rem'>{sub}</div>" if sub else ""
    return f"<div class='tile'><div class='big'>{value}</div><div class='cap'>{caption}</div>{extra}</div>"


def badge_html(b: dict) -> str:
    state = "on" if b["earned"] else "off"
    pct = int(100 * (b.get("progress") or (1 if b["earned"] else 0)))
    bar = "" if b["earned"] else f"<div class='bar'><i style='width:{pct}%'></i></div>"
    return f"""
<div class="badge {state}">
  <div class="ico">{b['icon']}</div>
  <div style="flex:1">
    <div class="ttl">{b['title']}</div>
    <div class="dsc">{b['description']}</div>
    {bar}
  </div>
</div>
"""


def _darken(hex_color: str, factor: float = 0.72) -> str:
    """Darken a #rrggbb colour for the card gradient's far stop."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def activity_panel(score: dict) -> str:
    """The whole 'can I go outside?' card as one HTML block.

    Built in one piece deliberately: Streamlit renders each st.* call as its own
    element, so a raw <div> opened in one markdown call cannot wrap widgets
    emitted by the next.
    """
    rows = "".join(
        f"<div class='act'><span>{a['name']}</span>"
        f"<span>{'&#10004;' if a['ok'] else '&#10008;'}</span></div>"
        for a in score["activities"]
    )
    pct = score["score"] or 0
    return f"""
<div class="panel">
  <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
              opacity:.65;font-weight:700">Can I go outside?</div>
  <div style="display:flex;align-items:baseline;gap:.5rem">
    <div style="font-size:2.6rem;font-weight:800">{pct}</div>
    <div style="opacity:.6;font-size:.9rem">/ 100</div>
  </div>
  <div style="font-weight:700;font-size:.9rem">{score['verdict']}</div>
  <div class="bar" style="margin:.5rem 0 .7rem"><i style="width:{pct}%"></i></div>
  {rows}
</div>
"""


def stat_strip(tiles_html: list[str], footer: str = "") -> str:
    """A row of tiles plus an optional footer, emitted as one block."""
    cells = "".join(f"<div style='flex:1'>{t}</div>" for t in tiles_html)
    foot = f"<div style='margin-top:.7rem'>{footer}</div>" if footer else ""
    return (f"<div class='panel'><div style='display:flex;gap:.6rem'>{cells}</div>"
            f"{foot}</div>")


# Registered once at import; every Plotly figure then inherits it, rather than
# each of the 15 call sites having to remember to set a background. Charts sit
# on their own light card so they stay legible whatever the sky is doing.
CHART_TEMPLATE = "aqi_sky"


def register_chart_template() -> str:
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates[CHART_TEMPLATE] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(255,255,255,.93)",
            plot_bgcolor="rgba(245,250,255,.85)",
            font=dict(color="#0a1929", size=12),
            title=dict(font=dict(color="#0a1929", size=15)),
            xaxis=dict(gridcolor="rgba(10,25,41,.10)", linecolor="rgba(10,25,41,.25)",
                       tickfont=dict(color="#0a1929"), zerolinecolor="rgba(10,25,41,.18)"),
            yaxis=dict(gridcolor="rgba(10,25,41,.10)", linecolor="rgba(10,25,41,.25)",
                       tickfont=dict(color="#0a1929"), zerolinecolor="rgba(10,25,41,.18)"),
            legend=dict(font=dict(color="#0a1929")),
            margin=dict(l=50, r=25, t=45, b=45),
            colorway=["#1565c0", "#e53935", "#00897b", "#8e24aa",
                      "#f57c00", "#546e7a"],
        )
    )
    pio.templates.default = f"plotly_white+{CHART_TEMPLATE}"
    return pio.templates.default

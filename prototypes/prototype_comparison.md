# GUI Prototype Comparison

This document compares minimal prototypes built using Dash and PySide6.
It measures basic load time and interaction latency on this container's hardware, then summarizes trade-offs.

## Measurements

| Framework | Load Time (s) | Interaction Latency (s) |
|-----------|---------------|------------------------|
| Dash      | 0.0129        | 0.0000                 |
| PySide6   | 0.0131        | 0.0001                 |

*Load time is measured from process start until the UI components are created.
Interaction latency measures a single programmatic button click.*

## Pros and Cons

### Dash
- **Pros:**
  - Web-based interface accessible through a browser.
  - Good ecosystem for interactive plots via Plotly.
  - Simple Python callback model.
- **Cons:**
  - Requires running a web server; user must access through HTTP.
  - Network stack introduces overhead for complex interactions.
  - Adds dependencies on `dash`, `plotly`, and `Flask`.

### PySide6
- **Pros:**
  - Native desktop widgets using the Qt framework.
  - Direct access to the underlying operating system and desktop features.
  - No web server required.
- **Cons:**
  - Large dependency footprint (~96 MB for Qt libraries plus system packages).
  - Requires several system libraries (e.g., `libgl1`, `libegl1`, `libxkbcommon`, `libxcb-cursor0`).
  - Qt programming model may be less familiar to developers used to web paradigms.

## Dependency Footprint

| Framework | Key Python Packages | Approx. Size |
|-----------|--------------------|--------------|
| Dash      | `dash`, `plotly`, `Flask` | ~10 MB Python packages |
| PySide6   | `PySide6` and Qt libraries | ~96 MB Python wheels + system libs |

## Developer Familiarity

- Existing code base uses Tkinter; neither Dash nor PySide6 are currently employed.
- Dash leverages web technologies (HTML/CSS/JS) which may be unfamiliar to a purely desktop-focused team.
- PySide6 requires knowledge of Qt's signal/slot architecture and event loop.

## Summary

Both prototypes show negligible startup and interaction times for a trivial view.
Dash offers easier deployment for multiple users via browser, while PySide6 provides a native desktop approach at the cost of larger dependencies and additional system libraries.

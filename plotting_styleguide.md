Here is the complete and precise primer on the duet_proba plotting style, including detailed explanations of the color scheme, stylistic choices, and the method for shading quantiles and confidence intervals.

-----

### **duet_proba Plotting Style: A Precise Primer**

The duet_proba library achieves its clean, publication-ready aesthetic through a specific set of rules defined in a Matplotlib `rcParams` dictionary. This guide breaks down these rules to help you replicate the style precisely.

#### **I. Machine-Readable Style Definition (JSON)**

This is the complete set of rules that defines the duet_proba plot style. You can use this dictionary directly with `matplotlib.pyplot.rcParams.update()` to apply the theme.

```json
{
  "theme_color": "#231F20",
  "style": {
    "lines.linewidth": 1.0,
    "lines.linestyle": "-",
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": "#231F20",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#231F20",
    "axes.linewidth": 0.8,
    "axes.grid": true,
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.labelcolor": "#231F20",
    "axes.prop_cycle": [ 
      "#4E79A7", 
      "#F28E2B", 
      "#E15759", 
      "#76B7B2", 
      "#59A14F", 
      "#EDC948", 
      "#B07AA1", 
      "#FF9DA7", 
      "#9C755F", 
      "#BAB0AC" 
    ], 
    "xtick.major.size": 2,
    "xtick.minor.size": 1,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "xtick.major.top": true,
    "xtick.major.bottom": true,
    "xtick.minor.top": true,
    "xtick.minor.bottom": true,
    "xtick.color": "#231F20",
    "ytick.major.size": 2,
    "ytick.minor.size": 1,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.6,
    "ytick.color": "#231F20",
    "ytick.major.left": true,
    "ytick.major.right": true,
    "ytick.minor.left": true,
    "ytick.minor.right": true,
    "grid.color": "#231F20",
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 1.0,
    "legend.frameon": false,
    "legend.edgecolor": "#231F20",
    "figure.figsize": [8, 5],
    "figure.dpi": 96,
    "figure.facecolor": "#FFFFFF",
    "figure.edgecolor": "#FFFFFF"
  }
}
```

-----

### **II. Detailed Style Explanation**

#### **🎨 Core Color Palette**

The theme is minimalistic, relying on a single base color for most elements and a vibrant cycle for data.

  * **Theme Color (`theme_color`)**: **`#231F20`**. This very dark gray is the cornerstone of the theme, used for all text, axes, tick marks, and grid lines. It provides strong contrast against the white background without being a harsh, pure black.
  * **Background Color**: **`#FFFFFF`** (white). Used for both the figure background (`figure.facecolor`) and the plotting area background (`axes.facecolor`).
  * **Data Color Cycle (`axes.prop_cycle`)**: When plotting multiple data series, Matplotlib cycles through these four distinct colors in order:
    1.  `#4E79A7` (Blue)
    2.  `#F28E2B` (Orange)
    3.  `#E15759` (Red)
    4.  `#76B7B2` (Teal)
    5.  `#59A14F` (Green)
    6.  `#EDC948` (Yellow)
    7.  `#B07AA1` (Purple)
    8.  `#FF9DA7` (Pink)
    9.  `#9C755F` (Brown)
    10. `#BAB0AC` (Gray)

#### **📐 Axes, Ticks, and Grid**

The plot's structure is defined by clean, thin lines and a full "boxed" appearance.

  * **Axes**: The border of the plot area (`axes.edgecolor`) is drawn with the dark `theme_color` and has a line width of **`0.8`**.
  * **Ticks**: To create a fully enclosed plot, major and minor ticks are enabled on all four sides (`top`, `bottom`, `left`, `right`). They are drawn with the `theme_color`.
  * **Grid**: A grid is enabled (`axes.grid: true`) to aid in reading values. It is styled to be unobtrusive:
      * **Style (`grid.linestyle`)**: Dotted (`:`).
      * **Width (`grid.linewidth`)**: Very thin (`0.4`).
      * **Color (`grid.color`)**: The dark `theme_color`.

#### **✒️ Fonts and Legend**

Text is kept clean, modern, and consistently sized.

  * **Font**: The style specifies a **`sans-serif`** font, with a base size of **`10`** for all text, including axes labels and legends.
  * **Legend**: The legend has no frame or background (`legend.frameon: false`), allowing it to float seamlessly over the plot area.

-----

### **III. Shading for Quantiles and Confidence Intervals**

duet_proba does **not** define separate hex codes for lighter or darker shades. Instead, it creates shaded regions for quantiles (like confidence intervals) by applying **alpha transparency** to a primary color. This is the standard and most effective way to show uncertainty bounds in Matplotlib.

#### **The Technique: `fill_between` with Alpha**

The core function used is `ax.fill_between()`. The process is as follows:

1.  A primary data line (e.g., the modeled return value) is plotted using a solid color from the color cycle (e.g., `#1f77b4`, Matplotlib's default blue, is often used in the diagnostic plots).
2.  The `ax.fill_between()` function is then called to create a filled polygon between the lower and upper confidence interval bounds.
3.  Crucially, this polygon is filled with the **exact same color** as the main line, but with its **`alpha`** (opacity) set to **`0.25`**. This makes the filled area appear as a light, transparent shade of the primary color.

#### **Precise Code Example:**

Here is how you would replicate this effect to plot a line with a 95% confidence interval, exactly as duet_proba does:

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
mean_line = np.sin(x)
upper_bound = mean_line + 0.3
lower_bound = mean_line - 0.3

# Use the pyextremes style
# plt.rcParams.update(pyextremes_rc)

fig, ax = plt.subplots()

# 1. Plot the main line (e.g., using the first color of the pyextremes cycle)
main_color = "#1771F1"
(line,) = ax.plot(x, mean_line, color=main_color, label="Modeled Value")

# 2. Add the shaded confidence interval region
ax.fill_between(
    x,
    lower_bound,
    upper_bound,
    color=line.get_color(),  # Use the same color as the line
    alpha=0.25,              # Set opacity to 25% to create the shade
    label="95% Confidence Interval"
)

ax.legend()
plt.show()
```
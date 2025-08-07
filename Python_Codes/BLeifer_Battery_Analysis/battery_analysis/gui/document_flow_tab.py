"""
Document Flow tab for the battery analysis GUI.

This module provides a tab that visualizes the document model structure
and flow of data between different documents in the MongoDB backend.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from battery_analysis.gui.custom_toolbar import CustomToolbar
from battery_analysis.utils import popout_figure
import networkx as nx
import logging
import threading
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

try:
    from .. import models
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")


class DocumentFlowTab(ttk.Frame):
    """Tab for visualizing document flow and relationships."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_view = "standard"  # 'standard', 'enhanced', or 'instances'
        self.instance_data = None
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the document flow tab."""
        # Top panel with controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # View selection
        ttk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(0, 5))

        self.view_var = tk.StringVar(value="standard")
        views = [
            ("Standard Schema", "standard"),
            ("Enhanced Materials Schema", "enhanced"),
            ("Your Database Instances", "instances"),
        ]

        for i, (text, value) in enumerate(views):
            ttk.Radiobutton(
                control_frame,
                text=text,
                variable=self.view_var,
                value=value,
                command=self.update_diagram,
            ).pack(side=tk.LEFT, padx=10)

        # Refresh button for instance data
        self.refresh_btn = ttk.Button(
            control_frame, text="Refresh Data", command=self.refresh_instance_data
        )
        self.refresh_btn.pack(side=tk.RIGHT, padx=10)

        # Main frame for the diagram
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a matplotlib figure for the diagram
        self.fig = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a navigation toolbar with editing support
        self.toolbar = CustomToolbar(self.canvas, self.main_frame)
        self.toolbar.update()

        # Button to open plot in a standalone window
        self.popout_btn = ttk.Button(
            self.main_frame,
            text="Open in Window",
            command=lambda: popout_figure(self.fig),
        )
        self.popout_btn.pack(anchor=tk.NE, padx=5, pady=5)

        # Info panel at the bottom
        info_frame = ttk.LabelFrame(self, text="Information")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.config(state=tk.DISABLED)

        # Add a scrollbar to the info text
        info_scrollbar = ttk.Scrollbar(
            self.info_text, orient=tk.VERTICAL, command=self.info_text.yview
        )
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=info_scrollbar.set)

        # Create initial diagram
        self.update_diagram()

    def update_diagram(self):
        """Update the document flow diagram based on the selected view."""
        view_type = self.view_var.get()
        self.current_view = view_type

        # Clear the figure
        self.fig.clear()

        # Draw the appropriate diagram
        if view_type == "standard":
            self.draw_standard_schema()
        elif view_type == "enhanced":
            self.draw_enhanced_schema()
        elif view_type == "instances":
            if self.instance_data is None:
                self.refresh_instance_data()
            else:
                self.draw_instance_diagram()

        # Update the canvas
        self.canvas.draw()

    def draw_standard_schema(self):
        """Draw the standard document schema diagram."""
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for each document type
        G.add_node("Sample", type="document", pos=(0.5, 0.8))
        G.add_node("TestResult", type="document", pos=(0.5, 0.5))
        G.add_node("CycleSummary", type="embedded", pos=(0.5, 0.2))

        # Add edges to show relationships
        G.add_edge("Sample", "Sample", label="parent", style="dashed")
        G.add_edge("Sample", "TestResult", label="tests", style="solid")
        G.add_edge("TestResult", "CycleSummary", label="cycles", style="solid")

        # Get positions
        pos = nx.get_node_attributes(G, "pos")

        # Create axis
        ax = self.fig.add_subplot(111)

        # Draw nodes with different styles based on type
        for node, node_attrs in G.nodes(data=True):
            if node_attrs.get("type") == "document":
                ax.add_patch(
                    Rectangle(
                        (pos[node][0] - 0.1, pos[node][1] - 0.05),
                        0.2,
                        0.1,
                        facecolor="skyblue",
                        edgecolor="black",
                        alpha=0.8,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (pos[node][0] - 0.1, pos[node][1] - 0.05),
                        0.2,
                        0.1,
                        facecolor="lightgreen",
                        edgecolor="black",
                        alpha=0.8,
                    )
                )

            ax.text(
                pos[node][0],
                pos[node][1],
                node,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                fontweight="bold",
            )

        # Draw edges with arrows and labels
        for u, v, edge_attrs in G.edges(data=True):
            # Handle self-loops specially
            if u == v:
                rad = 0.3
                ax.add_patch(
                    FancyArrowPatch(
                        posA=(pos[u][0] + 0.1, pos[u][1]),
                        posB=(pos[v][0] - 0.1, pos[v][1]),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>",
                        mutation_scale=15,
                        linewidth=1,
                        linestyle=edge_attrs.get("style", "solid"),
                        color="gray",
                    )
                )

                # Add label to self-loop
                ax.text(
                    pos[u][0] + 0.22,
                    pos[u][1] + 0.05,
                    edge_attrs.get("label", ""),
                    fontsize=10,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            else:
                # Calculate edge start and end points
                start_x, start_y = pos[u][0], pos[u][1] - 0.05  # bottom of node
                end_x, end_y = pos[v][0], pos[v][1] + 0.05  # top of node

                # Draw the edge
                ax.add_patch(
                    FancyArrowPatch(
                        posA=(start_x, start_y),
                        posB=(end_x, end_y),
                        arrowstyle="-|>",
                        mutation_scale=15,
                        linewidth=1,
                        linestyle=edge_attrs.get("style", "solid"),
                        color="gray",
                    )
                )

                # Add label
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                ax.text(
                    mid_x + 0.05,
                    mid_y,
                    edge_attrs.get("label", ""),
                    fontsize=10,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # Add property propagation flow
        prop_x = 0.8
        ax.add_patch(
            FancyArrowPatch(
                posA=(prop_x, 0.2),
                posB=(prop_x, 0.8),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2,
                linestyle="dashdot",
                color="green",
                label="Property Propagation",
            )
        )

        ax.text(
            prop_x + 0.1,
            0.5,
            "Property\nPropagation",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            color="green",
        )

        # Set axis limits and disable axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add title
        ax.set_title("Standard Document Schema")

        # Update info text
        self.update_info_text(
            """
        <b>Standard Document Schema</b>

        The standard schema uses three main document types:

        <b>Sample</b>: Represents a physical battery sample or cell
        - Can have a parent Sample (hierarchical relationship)
        - Contains references to TestResults
        - Stores aggregated metrics from tests (avg_capacity_retention, etc.)

        <b>TestResult</b>: Contains data from one battery test on a Sample
        - Linked to exactly one Sample
        - Contains a list of CycleSummary embedded documents
        - Stores summary metrics (cycle_count, initial_capacity, etc.)

        <b>CycleSummary</b>: Embedded document with per-cycle data
        - Contains charge/discharge capacities, coulombic efficiency, etc.
        - Embedded within TestResult documents (not independent)

        <b>Property Propagation</b>:
        When tests are added or updated, the metrics flow upward from CycleSummary to TestResult to Sample.
        If the Sample has a parent, metrics are further propagated up the hierarchy.
        """
        )

    def draw_enhanced_schema(self):
        """Draw the enhanced document schema for materials development."""
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for each document type
        G.add_node("EnhancedSample", type="document", pos=(0.5, 0.8))
        G.add_node("MaterialRole", type="embedded", pos=(0.25, 0.6))
        G.add_node("ProcessingStep", type="embedded", pos=(0.75, 0.6))
        G.add_node("TestResult", type="document", pos=(0.5, 0.4))
        G.add_node("CycleSummary", type="embedded", pos=(0.5, 0.2))

        # Add edges to show relationships
        G.add_edge("EnhancedSample", "EnhancedSample", label="parent", style="dashed")
        G.add_edge(
            "EnhancedSample",
            "EnhancedSample",
            label="derived_materials",
            style="dotted",
        )
        G.add_edge("EnhancedSample", "MaterialRole", label="components", style="solid")
        G.add_edge("MaterialRole", "EnhancedSample", label="material", style="solid")
        G.add_edge(
            "EnhancedSample", "ProcessingStep", label="processing_steps", style="solid"
        )
        G.add_edge("EnhancedSample", "TestResult", label="tests", style="solid")
        G.add_edge("TestResult", "CycleSummary", label="cycles", style="solid")

        # Get positions
        pos = nx.get_node_attributes(G, "pos")

        # Create axis
        ax = self.fig.add_subplot(111)

        # Draw nodes with different styles based on type
        for node, node_attrs in G.nodes(data=True):
            if node_attrs.get("type") == "document":
                ax.add_patch(
                    Rectangle(
                        (pos[node][0] - 0.12, pos[node][1] - 0.05),
                        0.24,
                        0.1,
                        facecolor="skyblue",
                        edgecolor="black",
                        alpha=0.8,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (pos[node][0] - 0.12, pos[node][1] - 0.05),
                        0.24,
                        0.1,
                        facecolor="lightgreen",
                        edgecolor="black",
                        alpha=0.8,
                    )
                )

            ax.text(
                pos[node][0],
                pos[node][1],
                node,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=11,
                fontweight="bold",
            )

        # Draw standard edges
        for u, v, edge_attrs in G.edges(data=True):
            # Skip self-loops for now
            if u == v:
                continue

            # Calculate edge start and end points
            start_x, start_y = pos[u][0], pos[u][1]
            end_x, end_y = pos[v][0], pos[v][1]

            # Adjust start points based on node position
            if end_y > start_y:  # Going up
                start_y += 0.05
                end_y -= 0.05
            elif end_y < start_y:  # Going down
                start_y -= 0.05
                end_y += 0.05

            if end_x > start_x:  # Going right
                start_x += 0.12
                end_x -= 0.12
            elif end_x < start_x:  # Going left
                start_x -= 0.12
                end_x += 0.12

            # Draw the edge
            ax.add_patch(
                FancyArrowPatch(
                    posA=(start_x, start_y),
                    posB=(end_x, end_y),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1,
                    linestyle=edge_attrs.get("style", "solid"),
                    color="gray",
                )
            )

            # Add label
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2

            # Offset label to avoid overlapping with arrow
            offset_x = 0
            offset_y = 0.03
            if abs(end_y - start_y) > abs(end_x - start_x):
                offset_x = 0.08
                offset_y = 0

            ax.text(
                mid_x + offset_x,
                mid_y + offset_y,
                edge_attrs.get("label", ""),
                fontsize=9,
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Draw self-loops (parent and derived_materials)
        # Parent relationship
        self._draw_self_loop(
            ax,
            pos["EnhancedSample"][0],
            pos["EnhancedSample"][1],
            0.35,
            "parent",
            "gray",
            "dashed",
        )

        # Derived materials relationship
        self._draw_self_loop(
            ax,
            pos["EnhancedSample"][0],
            pos["EnhancedSample"][1],
            -0.35,
            "derived_materials",
            "gray",
            "dotted",
        )

        # Add multi-directional property flow arrows
        # Upward propagation (traditional)
        self._draw_property_arrow(ax, 0.9, 0.3, 0.9, 0.7, "Upward\nPropagation", "blue")

        # Downward insights
        self._draw_property_arrow(ax, 0.1, 0.7, 0.1, 0.3, "Downward\nInsights", "green")

        # Set axis limits and disable axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add title
        ax.set_title("Enhanced Materials Development Schema")

        # Update info text
        self.update_info_text(
            """
        <b>Enhanced Materials Development Schema</b>

        The enhanced schema expands the standard model to track complete material development workflows:

        <b>EnhancedSample</b>: Represents any material, component, or assembly
        - Can have both traditional parent and component materials
        - Tracks both input components and derived products
        - Stores physical properties, classifications, and manufacturing metadata

        <b>MaterialRole</b>: Defines how a material is used in a composite or assembly
        - Specifies the role (cathode, anode, electrolyte, etc.)
        - Records proportion and processing parameters
        - Creates a network of material relationships

        <b>ProcessingStep</b>: Records manufacturing processes
        - Captures process parameters, equipment, and conditions
        - Links input materials to processing steps
        - Enables process-property relationship analysis

        <b>Multi-directional Property Propagation</b>:
        - Upward: Component properties influence assembly performance
        - Downward: Assembly performance provides insights about components
        - Lateral: Performance in one application informs others

        This schema enables tracking materials through complete development cycles from raw materials to finished cells and modules.
        """
        )

    def draw_instance_diagram(self):
        """Draw a diagram of actual document instances in the database."""
        if not self.instance_data:
            self.update_info_text(
                "No instance data available. Please connect to the database and click Refresh Data."
            )
            return

        # Extract data
        samples = self.instance_data.get("samples", [])
        tests = self.instance_data.get("tests", [])
        relationships = self.instance_data.get("relationships", [])

        if not samples:
            self.update_info_text("No samples found in the database.")
            return

        # Create a directed graph
        G = nx.DiGraph()

        # Add sample nodes
        for i, sample in enumerate(samples):
            # Calculate position - distribute samples in a grid
            row = i // 5
            col = i % 5
            pos_x = 0.1 + col * 0.2
            pos_y = 0.8 - row * 0.2

            G.add_node(
                sample["id"],
                type="sample",
                name=sample["name"],
                chemistry=sample.get("chemistry", ""),
                pos=(pos_x, pos_y),
            )

        # Add test nodes
        for i, test in enumerate(tests):
            # Position tests below their corresponding samples
            sample_id = test["sample_id"]

            # Find the position of the related sample
            if sample_id in G:
                sample_pos = G.nodes[sample_id]["pos"]
                # Offset slightly to avoid overlaps if multiple tests
                test_count = sum(1 for t in tests if t["sample_id"] == sample_id)
                test_idx = sum(1 for t in tests[:i] if t["sample_id"] == sample_id)

                offset_x = -0.05 * (test_count - 1) / 2 + 0.05 * test_idx
                pos_x = sample_pos[0] + offset_x
                pos_y = sample_pos[1] - 0.1
            else:
                # Fallback position if sample not found
                pos_x = 0.5
                pos_y = 0.3

            G.add_node(test["id"], type="test", name=test["name"], pos=(pos_x, pos_y))

            # Add edge from sample to test
            G.add_edge(sample_id, test["id"], relationship="tests")

        # Add parent-child relationships between samples
        for rel in relationships:
            if rel["type"] == "parent" and rel["source"] in G and rel["target"] in G:
                G.add_edge(rel["target"], rel["source"], relationship="parent")

        # Draw the graph
        ax = self.fig.add_subplot(111)
        pos = nx.get_node_attributes(G, "pos")

        # Draw nodes with different styles for samples and tests
        node_colors = []
        node_sizes = []
        for node, attrs in G.nodes(data=True):
            if attrs["type"] == "sample":
                color = "skyblue"
                size = 1200

                # Draw rectangular node for sample
                ax.add_patch(
                    Rectangle(
                        (pos[node][0] - 0.05, pos[node][1] - 0.025),
                        0.1,
                        0.05,
                        facecolor=color,
                        edgecolor="black",
                        alpha=0.8,
                    )
                )

                # Add text
                ax.text(
                    pos[node][0],
                    pos[node][1],
                    attrs["name"],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                    fontweight="bold",
                )

            else:  # test node
                color = "lightgreen"
                size = 800

                # Draw circular node for test
                ax.add_patch(
                    Circle(
                        (pos[node][0], pos[node][1]),
                        0.02,
                        facecolor=color,
                        edgecolor="black",
                        alpha=0.8,
                    )
                )

                # Add text
                ax.text(
                    pos[node][0],
                    pos[node][1] - 0.03,
                    attrs["name"],
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=7,
                )

            node_colors.append(color)
            node_sizes.append(size)

        # Draw edges
        for u, v, edge_attrs in G.edges(data=True):
            relationship = edge_attrs.get("relationship", "")

            # Style based on relationship type
            if relationship == "parent":
                style = "dashed"
                color = "blue"
            elif relationship == "tests":
                style = "solid"
                color = "black"
            else:
                style = "dotted"
                color = "gray"

            # Draw the edge
            ax.add_patch(
                FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                    mutation_scale=15,
                    linewidth=1,
                    linestyle=style,
                    color=color,
                )
            )

        # Add legend
        ax.add_patch(
            Rectangle((0.05, 0.05), 0.04, 0.02, facecolor="skyblue", edgecolor="black")
        )
        ax.text(0.1, 0.06, "Sample", fontsize=8, verticalalignment="center")

        ax.add_patch(
            Circle((0.05, 0.02), 0.01, facecolor="lightgreen", edgecolor="black")
        )
        ax.text(0.1, 0.02, "Test", fontsize=8, verticalalignment="center")

        # Set axis limits and disable axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add title
        ax.set_title("Database Document Instances")

        # Update info text
        self.update_info_text(
            f"""
        <b>Your Database Document Instances</b>

        This diagram shows the actual documents in your database:

        <b>Samples</b>: {len(samples)} in database
        <b>Tests</b>: {len(tests)} in database
        <b>Parent-Child Relationships</b>: {sum(1 for r in relationships if r['type'] == 'parent')}

        The diagram shows:
        - Rectangles: Sample documents
        - Circles: TestResult documents
        - Solid lines: Sample to TestResult relationships
        - Dashed lines: Parent to child Sample relationships

        This gives you a visual overview of your database structure and relationships.
        """
        )

    def refresh_instance_data(self):
        """Refresh the instance data from the database."""
        if not self.main_app.db_connected:
            messagebox.showwarning(
                "Not Connected", "Please connect to the database first."
            )
            return

        # Disable the refresh button during loading
        self.refresh_btn.config(state=tk.DISABLED)
        self.main_app.update_status("Loading document instances...")

        # Use a thread to avoid freezing the UI
        threading.Thread(target=self._load_instance_data_thread, daemon=True).start()

    def _load_instance_data_thread(self):
        """Thread function to load instance data from the database."""
        try:
            # Get samples
            samples = []
            for sample in models.Sample.objects().limit(
                30
            ):  # Limit to 30 for visualization clarity
                samples.append(
                    {
                        "id": str(sample.id),
                        "name": sample.name,
                        "chemistry": getattr(sample, "chemistry", ""),
                        "form_factor": getattr(sample, "form_factor", ""),
                        "has_parent": bool(getattr(sample, "parent", None)),
                    }
                )

            # Get tests
            tests = []
            for test in models.TestResult.objects().limit(
                50
            ):  # Limit to 50 for visualization clarity
                tests.append(
                    {
                        "id": str(test.id),
                        "name": test.name,
                        "sample_id": str(test.sample.id),
                        "cycle_count": getattr(test, "cycle_count", 0),
                    }
                )

            # Get relationships
            relationships = []
            # Parent-child relationships
            for sample in models.Sample.objects():
                if getattr(sample, "parent", None):
                    relationships.append(
                        {
                            "source": str(sample.parent.id),
                            "target": str(sample.id),
                            "type": "parent",
                        }
                    )

            # Store the data
            self.instance_data = {
                "samples": samples,
                "tests": tests,
                "relationships": relationships,
            }

            # Update the diagram if in instance view
            if self.current_view == "instances":
                self.main_app.queue.put(
                    {"type": "update_diagram", "callback": self.update_diagram}
                )

            # Update status
            self.main_app.update_status(
                f"Loaded {len(samples)} samples and {len(tests)} tests"
            )
            self.main_app.log_message(
                f"Loaded document instances: {len(samples)} samples, {len(tests)} tests"
            )

        except Exception as e:
            self.main_app.log_message(
                f"Error loading instance data: {str(e)}", logging.ERROR
            )
            self.main_app.update_status("Error loading instance data")

        finally:
            # Re-enable the refresh button
            self.refresh_btn.config(state=tk.NORMAL)

    def _draw_self_loop(self, ax, x, y, rad, label, color="gray", style="dashed"):
        """Helper to draw a self-loop on a node."""
        ax.add_patch(
            FancyArrowPatch(
                posA=(x + 0.1, y),
                posB=(x - 0.1, y),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1,
                linestyle=style,
                color=color,
            )
        )

        # Add label to self-loop
        angle = 90 if rad > 0 else -90  # Label position depends on loop direction
        ax.text(
            x,
            y + (0.1 if rad > 0 else -0.1),
            label,
            fontsize=9,
            rotation=angle,
            horizontalalignment="center",
            verticalalignment="center",
        )

    def _draw_property_arrow(self, ax, x1, y1, x2, y2, label, color="green"):
        """Helper to draw a property propagation arrow."""
        ax.add_patch(
            FancyArrowPatch(
                posA=(x1, y1),
                posB=(x2, y2),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2,
                linestyle="dashdot",
                color=color,
            )
        )

        # Add label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x + 0.05,
            mid_y,
            label,
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            color=color,
        )

    def update_info_text(self, text):
        """Update the information text panel."""
        # Enable editing
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        # Process simple HTML-like formatting
        formatted_text = ""

        for line in text.split("\n"):
            line = line.strip()

            # Process HTML-like tags
            while "<b>" in line and "</b>" in line:
                start = line.find("<b>")
                end = line.find("</b>")

                # Add text before the tag
                formatted_text += line[:start]

                # Add the bold text
                bold_text = line[start + 3 : end]
                formatted_text += bold_text

                # Store this range for applying bold tag later
                length = len(bold_text)
                self.info_text.insert(tk.END, formatted_text)
                self.info_text.tag_add("bold", f"end-{length}c", "end")
                formatted_text = ""

                # Continue with rest of line
                line = line[end + 4 :]

            # Add any remaining text from this line
            formatted_text += line + "\n"

        # Insert the remaining text
        self.info_text.insert(tk.END, formatted_text)

        # Configure styles
        self.info_text.tag_configure("bold", font=("Arial", 10, "bold"))

        # Disable editing
        self.info_text.config(state=tk.DISABLED)

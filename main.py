import os
import hashlib
import warnings
import webbrowser
from tkinter import *
from tkinter import filedialog, ttk, messagebox, END
import community as community_louvain
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community import girvan_newman

class SocialNetworkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Social Networks Analysis Tool")
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.current_graph_type = "undirected"
        self.node_attributes = {}
        self.edge_attributes = {}
        self.original_graph = None
        self.original_directed_graph = None
        self.original_node_attributes = {}
        self.original_edge_attributes = {}

        # Suppress PyVis warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        # Themes
        self.dark_theme = {
            "bg_main": "#121212",
            "bg_secondary": "#1E1E1E",
            "button_bg": "#BB86FC",
            "button_text_color": "#121212",
            "text_color": "#F5F5F5",
            "error_color": "#CF6679",
        }
        self.light_theme = {
            "bg_main": "#D9D9D9",
            "bg_secondary": "#D9D9D9",
            "button_bg": "#00B96D",
            "button_text_color": "#121212",
            "text_color": "#000000",
            "error_color": "#CF6679",
        }
        self.current_theme = self.dark_theme

        self.setup_ui()

    def setup_ui(self):
        # Main frame with scrollable content
        self.main_frame = Frame(self.root, padx=10, pady=10, bg=self.current_theme["bg_main"])
        self.main_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # Scrollbar and canvas for scrolling
        self.canvas = Canvas(self.main_frame, bg=self.current_theme["bg_main"], highlightthickness=0)
        self.scrollbar = Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill=BOTH, expand=True)

        # Frame inside canvas
        self.control_frame = Frame(self.canvas, padx=10, pady=10, bg=self.current_theme["bg_main"])
        self.canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        self.control_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Mousewheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))  # Linux

        # Visualization frame
        self.visualization_frame = Frame(self.root, padx=10, pady=10, bg=self.current_theme["bg_secondary"])
        self.visualization_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # --------- UI Content ---------
        self.title_frame = Frame(self.control_frame, bg=self.current_theme["bg_main"])
        self.title_frame.grid(row=0, column=0, columnspan=2, sticky='ew')

        Label(self.title_frame, text="Load Network Data", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).pack(side=LEFT, pady=(0, 10))

        # Theme toggle using Checkbutton (Switch-like)
        self.theme_var = IntVar(value=0)  # 0 = Dark Theme, 1 = Light Theme
        self.theme_toggle = Checkbutton(self.title_frame, text="Theme", variable=self.theme_var,
                                        command=self.toggle_theme,
                                        bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                                        selectcolor=self.current_theme["button_bg"])
        self.theme_toggle.pack(side=RIGHT, pady=(0, 10), padx=(10, 0))

        # Buttons with button_text_color applied
        Button(self.control_frame, text="Load Nodes CSV", command=self.load_nodes, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=1, column=0,
                                                                                                    pady=5, sticky='ew')
        Button(self.control_frame, text="Load Edges CSV", command=self.load_edges, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=1, column=1,
                                                                                                    pady=5, sticky='ew')

        Button(self.control_frame, text="Reset Data", command=self.reset_data, width=20,
               bg=self.current_theme["error_color"], fg="white").grid(row=2, column=0, columnspan=2, pady=(10, 10),
                                                                      sticky='ew')

        Label(self.control_frame, text="Graph Type:", font=('Arial', 14),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=3, column=0, pady=5,
                                                                                          sticky='w')
        self.graph_type = StringVar(value="undirected")
        Radiobutton(self.control_frame, text="Undirected", variable=self.graph_type, value="undirected",
                    command=self.on_graph_type_changed, bg=self.current_theme["bg_main"],
                    fg=self.current_theme["text_color"], selectcolor=self.current_theme["bg_main"]).grid(row=4,
                                                                                                         column=0,
                                                                                                         sticky='w')
        Radiobutton(self.control_frame, text="Directed", variable=self.graph_type, value="directed",
                    command=self.on_graph_type_changed, bg=self.current_theme["bg_main"],
                    fg=self.current_theme["text_color"], selectcolor=self.current_theme["bg_main"]).grid(row=4,
                                                                                                         column=1,
                                                                                                         sticky='w')

        # Visualization Options
        Label(self.control_frame, text="Visualization Options", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=5, column=0, columnspan=2,
                                                                                          pady=(20, 10), sticky='w')
        # Node Shape Attribute
        self.node_shape_attr = self.create_combobox_with_label("Node Shape Attribute:",
                                                               ["dot", "square", "triangle", "star"], 6)
        self.node_shape_attr.set("dot")

        # Edge Style
        self.edge_style_attr = self.create_combobox_with_label("Edge Style:", ["solid", "dashed", "dotted"], 7)
        self.edge_style_attr.set("solid")

        # Layout Algorithm
        self.layout_algo = self.create_combobox_with_label("Layout Algorithm:",
                                                           ["force-directed", "hierarchical", "circular", "random"], 8)
        self.layout_algo.set("force-directed")

        Label(self.control_frame, text="Node Size Attribute:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=9, column=0, pady=5, sticky='w')
        self.node_size_attr = ttk.Combobox(self.control_frame, width=18)
        self.node_size_attr.grid(row=9, column=1, pady=5, sticky='ew')

        Label(self.control_frame, text="Node Color Attribute:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=10, column=0, pady=5, sticky='w')
        self.node_color_attr = ttk.Combobox(self.control_frame, width=18)
        self.node_color_attr.grid(row=10, column=1, pady=5, sticky='ew')

        # Link Analysis
        Label(self.control_frame, text="Link Analysis", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=11, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')
        Button(self.control_frame, text="Run PageRank", command=self.run_pagerank, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=12, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')
        Button(self.control_frame, text="Run Betweenness", command=self.run_betweenness, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=13, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')

        # Community Detection
        Label(self.control_frame, text="Community Detection", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=14, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')
        self.community_algo = self.create_combobox_with_label(
            "Community Algorithm:", ["Louvain", "Girvan-Newman", "Both"], 15
        )
        self.community_algo.set("Louvain")
        Button(self.control_frame, text="Detect Communities", command=self.detect_communities, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=16, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=10,
                                                                                                    sticky='ew')

        # Filtering
        Label(self.control_frame, text="Filtering Options", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=17, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')

        self.selected_community = self.create_combobox_with_label("Filter by Community:", self.get_community_list(), 18)
        Button(self.control_frame, text="Apply Community Filter", command=self.apply_community_filter, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=19, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')

        self.centrality_type = self.create_combobox_with_label("Filter by Centrality:", ["degree", "betweenness", "eigenvector"], 20)

        Label(self.control_frame, text="Min Value:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=21, column=0, pady=5, sticky='w')
        self.min_centrality = Entry(self.control_frame, width=18)
        self.min_centrality.grid(row=21, column=1, pady=5, sticky='ew')

        Label(self.control_frame, text="Max Value:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=22, column=0, pady=5, sticky='w')
        self.max_centrality = Entry(self.control_frame, width=18)
        self.max_centrality.grid(row=22, column=1, pady=5, sticky='ew')

        Button(self.control_frame, text="Apply Centrality Filter", command=self.apply_centrality_filter, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=23, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=10,
                                                                                                    sticky='ew')

        # Visualize Network
        Button(self.control_frame, text="Visualize Network", command=self.visualize_network,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"], font=('Arial', 12, 'bold')).grid(row=24, column=0,
                                                                                                columnspan=2, pady=20,
                                                                                                sticky='ew')

        # Metrics Display
        self.metrics_text = Text(
            self.visualization_frame,
            width=80,
            height=30,
            bg=self.current_theme["bg_secondary"],
            fg=self.current_theme["text_color"],
            font=('Arial', 12),
            wrap="word",
            padx=10,
            pady=10,
            bd=1,
            highlightbackground="white" if self.current_theme == self.dark_theme else "black",
            highlightthickness=1,
        )
        self.metrics_text.pack(expand=True, fill=BOTH, padx=10, pady=10)

    def create_combobox_with_label(self, label_text, values, row_num):
        Label(self.control_frame, text=label_text, bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=row_num, column=0, pady=5, sticky='w')
        combobox = ttk.Combobox(self.control_frame, values=values, width=18)
        combobox.grid(row=row_num, column=1, pady=5, sticky='ew')
        return combobox

    def toggle_theme(self):
        # Switch between light and dark themes based on Checkbutton state
        if self.current_theme == self.dark_theme:
            self.current_theme = self.light_theme
        else:
            self.current_theme = self.dark_theme

        # Update UI elements with the new theme colors
        self.update_ui_theme()

    def update_ui_theme(self):
        # Update the main frame background color
        self.main_frame.config(bg=self.current_theme["bg_main"])

        # Update the canvas and scrollbar
        self.canvas.config(bg=self.current_theme["bg_main"])
        self.scrollbar.config(bg=self.current_theme["bg_main"])

        # Update the control frame and its widgets (except ComboBox, input fields, and reset button)
        self.control_frame.config(bg=self.current_theme["bg_main"])
        for widget in self.control_frame.winfo_children():
            if isinstance(widget, Label):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"])
            elif isinstance(widget, Button) and widget.cget("text") != "Reset Data":  # Exclude reset button
                widget.config(bg=self.current_theme["button_bg"],
                              fg=self.current_theme["button_text_color"])
            elif isinstance(widget, Checkbutton):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                              selectcolor=self.current_theme["button_bg"])
            elif isinstance(widget, Radiobutton):  # Update Radiobutton background as well
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                              selectcolor=self.current_theme["bg_main"])

        # Update the visualization frame
        self.visualization_frame.config(bg=self.current_theme["bg_secondary"])

        # Update the text area background and text color
        self.metrics_text.config(bg=self.current_theme["bg_secondary"], fg=self.current_theme["text_color"])

        # Update the theme toggle button (Checkbutton)
        self.theme_toggle.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                                 selectcolor=self.current_theme["button_bg"])

        # Update the title_frame background color and the Label inside it
        self.title_frame.config(bg=self.current_theme["bg_main"])
        for widget in self.title_frame.winfo_children():
            if isinstance(widget, Label):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"])

    def on_graph_type_changed(self):
        self.current_graph_type = self.graph_type.get()
        self.calculate_basic_metrics()

    def reset_data(self):
        # Restore original graphs
        if self.original_graph is not None:
            self.graph = self.original_graph.copy()
        if self.original_directed_graph is not None:
            self.directed_graph = self.original_directed_graph.copy()
        # Restore node and edge attributes
        self.node_attributes = self.original_node_attributes.copy()
        self.edge_attributes = self.original_edge_attributes.copy()
        # Update UI elements
        self.update_attribute_comboboxes()
        self.calculate_basic_metrics()
        self.metrics_text.insert(END, "\nData reset to original uploaded files.\n")

    def load_nodes(self):
        filepath = filedialog.askopenfilename(title="Select Nodes CSV File", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                df = pd.read_csv(filepath)
                df.columns = [col.lower() for col in df.columns]

                if 'id' not in df.columns:
                    messagebox.showerror("Error", "CSV file must contain an 'id' column (case insensitive)")
                    return

                self.original_node_attributes = df.set_index('id').to_dict('index')
                self.node_attributes = self.original_node_attributes.copy()
                self.update_attribute_comboboxes()
                self.metrics_text.insert(END, f"Loaded {len(df)} nodes from {filepath}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading nodes: {str(e)}")

    def load_edges(self):
        filepath = filedialog.askopenfilename(title="Select Edges CSV File", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                df = pd.read_csv(filepath)
                df.columns = [col.lower() for col in df.columns]

                if 'source' not in df.columns or 'target' not in df.columns:
                    messagebox.showerror("Error", "CSV file must contain 'Source' and 'Target' columns (case insensitive)")
                    return

                if 'weight' not in df.columns:
                    df['weight'] = 1

                df = df.groupby(['source', 'target'], as_index=False).agg({'weight': 'sum'})

                # Handle undirected vs directed graphs for edge attributes
                if self.graph_type.get() == "undirected":
                    # Sort source and target to ensure consistent order
                    df[['source', 'target']] = df.apply(lambda x: sorted([x['source'], x['target']]), axis=1, result_type='expand')
                    # Regroup after sorting
                    df = df.groupby(['source', 'target'], as_index=False).agg({'weight': 'sum'})

                # Create original graphs
                self.original_graph = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True)
                self.original_directed_graph = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
                self.graph = self.original_graph.copy()
                self.directed_graph = self.original_directed_graph.copy()

                # Edge attributes (including weight)
                self.original_edge_attributes = df.set_index(['source', 'target']).to_dict('index')
                self.edge_attributes = self.original_edge_attributes.copy()

                self.current_graph_type = self.graph_type.get()
                self.update_attribute_comboboxes()
                self.metrics_text.insert(END, f"Loaded {len(df)} edges from {filepath}\n")
                self.calculate_basic_metrics()

            except Exception as e:
                messagebox.showerror("Error", f"Error loading edges: {str(e)}")

    def update_attribute_comboboxes(self):
        if self.node_attributes:
            attributes = list(next(iter(self.node_attributes.values())).keys())
            self.node_size_attr['values'] = attributes
            self.node_color_attr['values'] = attributes
            if attributes:
                self.node_size_attr.set(attributes[0])
                self.node_color_attr.set(attributes[0])

    def calculate_basic_metrics(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        self.metrics_text.delete(1.0, END)  # Clear previous content

        if g.number_of_nodes() == 0:
            self.metrics_text.insert(END, "Graph is empty - no metrics to calculate\n")
            return

        self.metrics_text.insert(END, "\n=== Network Metrics ===\n")
        self.metrics_text.insert(END, f"Number of nodes: {g.number_of_nodes()}\n")
        self.metrics_text.insert(END, f"Number of edges: {g.number_of_edges()}\n")

        # Check for strong connectivity in directed graphs
        if self.current_graph_type == "directed":
            try:
                if not nx.is_strongly_connected(g):
                    self.metrics_text.insert(END,
                                             "Warning: The directed graph is not strongly connected. Some metrics may not be calculable.\n")
            except nx.NetworkXPointlessConcept:
                pass

        # Average Shortest Path Length
        try:
            if self.current_graph_type == "directed":
                if nx.is_strongly_connected(g):
                    avg_path = nx.average_shortest_path_length(g)
                    self.metrics_text.insert(END, f"Average shortest path length (directed): {avg_path:.4f}\n")
                else:
                    self.metrics_text.insert(END,
                                             "Cannot calculate average shortest path length: directed graph is not strongly connected.\n")
            else:
                if nx.is_connected(g):
                    avg_path = nx.average_shortest_path_length(g)
                    self.metrics_text.insert(END, f"Average shortest path length: {avg_path:.4f}\n")
                else:
                    self.metrics_text.insert(END,
                                             "Cannot calculate average shortest path length: graph is not connected.\n")
        except nx.NetworkXPointlessConcept:
            self.metrics_text.insert(END, "Graph is empty - cannot calculate average shortest path length.\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Error calculating average shortest path length: {str(e)}\n")

        # Average Clustering Coefficient
        try:
            self.metrics_text.insert(END, f"Average clustering coefficient: {nx.average_clustering(g):.4f}\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Could not calculate clustering coefficient: {str(e)}\n")

        # Degree Assortativity Coefficient (with warning suppression)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                assortativity = nx.degree_assortativity_coefficient(g)
                if np.isnan(assortativity):
                    self.metrics_text.insert(END,
                                             "Degree assortativity coefficient: Could not compute (invalid division)\n")
                else:
                    self.metrics_text.insert(END, f"Degree assortativity coefficient: {assortativity:.4f}\n")
        except Exception as e:
            self.metrics_text.insert(END, f"Could not calculate assortativity: {str(e)}\n")

        # Degree Distribution
        degree_sequence = [d for n, d in g.degree()]
        degree_count = {i: degree_sequence.count(i) for i in set(degree_sequence)}
        self.metrics_text.insert(END, f"Degree Distribution: {degree_count}\n")

        # Degree Centrality
        try:
            degree_centrality = nx.degree_centrality(g)
            self.metrics_text.insert(END, f"Degree Centrality: {degree_centrality}\n")
        except Exception as e:
            self.metrics_text.insert(END, f"Could not calculate degree centrality: {str(e)}\n")

        # Betweenness Centrality
        try:
            betweenness_centrality = nx.betweenness_centrality(g)
            self.metrics_text.insert(END, f"Betweenness Centrality: {betweenness_centrality}\n")
        except Exception as e:
            self.metrics_text.insert(END, f"Could not calculate betweenness centrality: {str(e)}\n")

        # Eigenvector Centrality (with Katz fallback)
        try:
            if self.current_graph_type == "directed":
                if not nx.is_strongly_connected(g):
                    self.metrics_text.insert(END, "Eigenvector centrality unavailable: graph not strongly connected\n")
                    self.metrics_text.insert(END, "Calculating Katz centrality instead...\n")
                    katz = nx.katz_centrality(g, alpha=0.1, max_iter=1000)
                    self.metrics_text.insert(END, f"Katz Centrality: {katz}\n")
                else:
                    eigenvector = nx.eigenvector_centrality(g, max_iter=1000)
                    self.metrics_text.insert(END, f"Eigenvector Centrality: {eigenvector}\n")
            else:  # undirected
                if not nx.is_connected(g):
                    self.metrics_text.insert(END, "Eigenvector centrality unavailable: graph not connected\n")
                    self.metrics_text.insert(END, "Calculating Katz centrality instead...\n")
                    katz = nx.katz_centrality(g, alpha=0.1, max_iter=1000)
                    self.metrics_text.insert(END, f"Katz Centrality: {katz}\n")
                else:
                    eigenvector = nx.eigenvector_centrality(g, max_iter=1000)
                    self.metrics_text.insert(END, f"Eigenvector Centrality: {eigenvector}\n")
        except nx.PowerIterationFailedConvergence:
            self.metrics_text.insert(END, "Eigenvector failed to converge - using Katz centrality\n")
            katz = nx.katz_centrality(g, alpha=0.1)
            self.metrics_text.insert(END, f"Katz Centrality: {katz}\n")
        except Exception as e:
            self.metrics_text.insert(END, f"Centrality calculation error: {str(e)}\n")

    def visualize_network(self):
        if self.current_graph_type == "undirected":
            g = self.graph.copy()
        else:
            g = self.directed_graph.copy()

        if len(g.nodes()) == 0:
            messagebox.showwarning("Warning", "No network data to visualize")
            return

        try:
            # Create network with directed flag
            net = Network(notebook=True, height="750px", width="100%",
                          directed=(self.current_graph_type == "directed"),
                          cdn_resources='remote')

            # Add nodes with attributes
            for node in g.nodes():
                node_attrs = {"label": str(node), "title": str(node)}

                if self.node_attributes and node in self.node_attributes:
                    for attr, value in self.node_attributes[node].items():
                        node_attrs[attr] = value

                        if attr == self.node_size_attr.get():
                            node_attrs["size"] = float(value) * 5 if str(value).replace('.', '').isdigit() else 10
                        if attr == self.node_color_attr.get():
                            node_attrs["color"] = self.value_to_color(value)
                        if attr == "shape":
                            if value in ['dot', 'square', 'triangle', 'star']:
                                node_attrs["shape"] = value
                            else:
                                node_attrs["shape"] = 'dot'

                # Get the selected shape from combobox
                selected_shape = self.node_shape_attr.get()
                if selected_shape in ['dot', 'square', 'triangle', 'star']:
                    node_attrs["shape"] = selected_shape
                else:
                    node_attrs["shape"] = 'dot'

                net.add_node(node, **node_attrs)

            # Add edges with attributes
            for edge in g.edges():
                edge_attrs = {}
                u, v = edge

                # Check both edge directions for undirected graphs
                if self.current_graph_type == "undirected":
                    key = (u, v) if (u, v) in self.edge_attributes else (v, u)
                else:
                    key = (u, v)

                if key in self.edge_attributes:
                    for attr, value in self.edge_attributes[key].items():
                        edge_attrs[attr] = value
                        if attr == "style":
                            edge_attrs["dashes"] = True if value == "dashed" else False
                        if attr == "weight":
                            edge_attrs["width"] = float(value) / 10 if value else 1
                        if attr == "color":
                            edge_attrs["color"] = self.value_to_color(value)
                        if attr == "width":
                            edge_attrs["width"] = float(value)

                selected_style = self.edge_style_attr.get()
                if selected_style == "dashed":
                    edge_attrs["dashes"] = True
                elif selected_style == "dotted":
                    edge_attrs["dashes"] = [2, 5]
                else:
                    edge_attrs["dashes"] = False

                # Add arrow configuration for directed graphs
                if self.current_graph_type == "directed":
                    edge_attrs["arrows"] = "to"
                    edge_attrs["smooth"] = False

                net.add_edge(edge[0], edge[1], **edge_attrs)

            # Apply layout
            layout = self.layout_algo.get()
            if layout == "force-directed":
                net.force_atlas_2based(gravity=-50)
            elif layout == "hierarchical":
                net.set_options("""
                {
                    "layout": {
                        "hierarchical": {
                            "enabled": true,
                            "direction": "UD",
                            "sortMethod": "directed"
                        }
                    }
                }
                """)
            elif layout == "circular":
                net.set_options("""
                {
                    "layout": {
                        "randomSeed": 42,
                        "improvedLayout": false
                    },
                    "physics": {
                        "hierarchicalRepulsion": {
                            "centralGravity": 0.0,
                            "springLength": 200,
                            "springConstant": 0.01,
                            "nodeDistance": 100,
                            "damping": 0.09
                        },
                        "minVelocity": 0.75,
                        "solver": "repulsion"
                    }
                }
                """)
            elif layout == "random":
                net.set_options("""
                {
                    "layout": {
                        "randomSeed": 42
                    }
                }
                """)

            # Save and show
            output_file = "network_visualization.html"
            net.show(output_file)
            webbrowser.open('file://' + os.path.realpath(output_file))

        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")

    def value_to_color(self, value):


        # Handle None or empty values
        if value is None:
            return "hsl(0, 0%, 50%)"  # Gray as fallback

        # Convert numeric values to strings consistently
        if isinstance(value, (int, float)):
            # Round floats to 4 decimal places to ensure consistent colors
            value = f"{float(value):.4f}"
        else:
            value = str(value)

        # Generate hash - using sha256 for better distribution than md5
        hash_val = int(hashlib.sha256(value.encode()).hexdigest(), 16)

        # Calculate HSL values
        h = hash_val % 360  # Hue (0-359)
        s = 80 + (hash_val % 21)  # Saturation (80-100%) - more vibrant colors
        l = 40 + (hash_val % 21)  # Lightness (40-60%) - better contrast

        return f"hsl({h}, {s}%, {l}%)"

    def apply_centrality_filter(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to filter")
            return

        centrality_type = self.centrality_type.get()
        min_val = float(self.min_centrality.get()) if self.min_centrality.get() else 0
        max_val = float(self.max_centrality.get()) if self.max_centrality.get() else float('inf')

        try:
            if centrality_type == "degree":
                centrality = nx.degree_centrality(g)
            elif centrality_type == "betweenness":
                centrality = nx.betweenness_centrality(g)
            elif centrality_type == "eigenvector":
                # Check graph connectivity first
                if self.current_graph_type == "directed":
                    if not nx.is_strongly_connected(g):
                        messagebox.showwarning("Cannot Calculate",
                                               "Eigenvector centrality requires strongly connected graphs.\n"
                                               "Please select a different centrality measure.")
                        return
                else:
                    if not nx.is_connected(g):
                        messagebox.showwarning("Cannot Calculate",
                                               "Eigenvector centrality requires connected graphs.\n"
                                               "Please select a different centrality measure.")
                        return

                # If we get here, the graph is properly connected
                try:
                    centrality = nx.eigenvector_centrality(g, max_iter=500)
                except nx.PowerIterationFailedConvergence:
                    messagebox.showwarning("Calculation Failed",
                                           "Eigenvector centrality failed to converge.\n"
                                           "Please try degree or betweenness centrality instead.")
                    return
            else:
                messagebox.showwarning("Warning", "Please select a valid centrality type")
                return

            filtered_nodes = [n for n, c in centrality.items() if min_val <= c <= max_val]

            if not filtered_nodes:
                messagebox.showwarning("Warning", "No nodes match the filter criteria")
                return

            filtered_graph = g.subgraph(filtered_nodes)

            if self.current_graph_type == "undirected":
                self.graph = filtered_graph
            else:
                self.directed_graph = filtered_graph

            self.metrics_text.insert(END,
                                     f"\nApplied {centrality_type} centrality filter: "
                                     f"{len(filtered_nodes)}/{g.number_of_nodes()} nodes remain\n")
            self.calculate_basic_metrics()

        except Exception as e:
            messagebox.showerror("Error",
                                 f"Failed to calculate centrality:\n{str(e)}\n"
                                 "Please try a different centrality measure.")
    def apply_community_filter(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to filter")
            return

        selected_community = self.selected_community.get()

        if not selected_community:
            messagebox.showwarning("Warning", "Please select a community to filter by")
            return

        # Ensure selected_community is treated as an integer if communities are integers
        try:
            selected_community = int(selected_community)  # Convert to int if communities are integers
        except ValueError:
            pass  # If it's not an integer, it may be a string (you can handle accordingly)

        try:
            # Filter nodes based on selected community
            filtered_nodes = [n for n in g.nodes() if
                              self.node_attributes.get(n, {}).get('community') == selected_community]

            if not filtered_nodes:
                messagebox.showwarning("Warning", "No nodes match the selected community")
                return

            # Create subgraph with filtered nodes
            filtered_graph = g.subgraph(filtered_nodes)

            if self.current_graph_type == "undirected":
                self.graph = filtered_graph
            else:
                self.directed_graph = filtered_graph

            # Display filtering results
            self.metrics_text.insert(END,
                                     f"\nApplied community filter: {len(filtered_nodes)}/{g.number_of_nodes()} nodes remain\n")
            self.calculate_basic_metrics()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during community filtering: {str(e)}")

    def detect_communities(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph.to_undirected()

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to analyze")
            return

        # Get selected algorithm
        algo = self.community_algo.get()

        try:
            # Initialize variables for storing community results
            louvain_partition = None
            louvain_modularity = None
            louvain_communities = None
            girvan_communities = None
            girvan_edges_removed = None

            # Run Louvain Community Detection if selected
            if algo == "Louvain" or algo == "Both":
                louvain_partition = community_louvain.best_partition(g)
                louvain_modularity = community_louvain.modularity(louvain_partition, g)
                louvain_communities = {comm_id: [] for comm_id in louvain_partition.values()}
                for node, comm_id in louvain_partition.items():
                    louvain_communities[comm_id].append(node)

            # Run Girvan-Newman Community Detection if selected
            if algo == "Girvan-Newman" or algo == "Both":
                comp = girvan_newman(g)
                girvan_communities = tuple(sorted(c) for c in next(comp))
                # Calculate edges removed
                original_edges = len(g.edges())
                remaining_edges = sum(g.subgraph(comm).number_of_edges() for comm in girvan_communities)
                girvan_edges_removed = original_edges - remaining_edges

            # Display comparison results in GUI
            self.metrics_text.insert(END, "\nCommunity Detection Comparison:\n")

            # Function to calculate community size distribution
            def get_community_sizes(community_dict):
                return [len(nodes) for nodes in community_dict.values()]

            # Function to calculate conductance
            def conductance(graph, community_nodes):
                cut_size = sum(1 for u in community_nodes for v in graph.neighbors(u) if v not in community_nodes)
                volume = len(community_nodes) * (len(graph.nodes()) - len(community_nodes))
                return cut_size / volume if volume > 0 else 0

            # Display Louvain results if available
            if louvain_partition:
                community_sizes = get_community_sizes(louvain_communities)
                self.metrics_text.insert(END, f"\nLouvain Algorithm Results:\n")
                self.metrics_text.insert(END, f"Number of communities detected: {len(louvain_communities)}\n")
                self.metrics_text.insert(END, f"Community Size Distribution: {community_sizes}\n")

                for comm_id, nodes in louvain_communities.items():
                    community_conductance = conductance(g, nodes)
                    self.metrics_text.insert(END, f"Conductance of community {comm_id}: {community_conductance:.4f}\n")
                    self.metrics_text.insert(END, f"Nodes in community {comm_id}: {', '.join(map(str, nodes))}\n")

                # Add community as node attribute for visualization
                for node in g.nodes():
                    self.node_attributes[node] = {'community': louvain_partition[node]}

            # Display Girvan-Newman results if available
            if girvan_communities:
                self.metrics_text.insert(END, f"\nGirvan-Newman Algorithm Results:\n")
                self.metrics_text.insert(END, f"Number of communities detected: {len(girvan_communities)}\n")
                self.metrics_text.insert(END, f"Number of edges removed: {girvan_edges_removed}\n")

                girvan_community_sizes = get_community_sizes({i: comm for i, comm in enumerate(girvan_communities)})
                self.metrics_text.insert(END,
                                         f"Community Size Distribution (Girvan-Newman): {girvan_community_sizes}\n")

                for comm_id, nodes in enumerate(girvan_communities):
                    community_conductance = conductance(g, nodes)
                    self.metrics_text.insert(END, f"Conductance of community {comm_id}: {community_conductance:.4f}\n")
                    self.metrics_text.insert(END, f"Nodes in community {comm_id}: {', '.join(map(str, nodes))}\n")

                for i, comm in enumerate(girvan_communities):
                    for node in comm:
                        self.node_attributes[node] = {'community': i}

            # Clustering Evaluation Section
            self.metrics_text.insert(END, "\nClustering Evaluation:\n")

            # 1. Internal: Louvain Modularity
            if louvain_modularity:
                self.metrics_text.insert(END, f"Louvain Modularity: {louvain_modularity:.4f}\n")

            # 2. Internal: Girvan-Newman Modularity
            if girvan_communities:
                girvan_partition = {node: next(i for i, c in enumerate(girvan_communities) if node in c) for node in
                                    g.nodes()}
                girvan_modularity = community_louvain.modularity(girvan_partition, g)
                self.metrics_text.insert(END, f"Girvan-Newman Modularity: {girvan_modularity:.4f}\n")

            # 3. External: Normalized Mutual Information (NMI)
            if louvain_partition and girvan_communities:
                louvain_labels = [louvain_partition[node] for node in g.nodes()]
                girvan_labels = [next(i for i, c in enumerate(girvan_communities) if node in c) for node in g.nodes()]
                nmi_score = normalized_mutual_info_score(louvain_labels, girvan_labels)
                self.metrics_text.insert(END,
                                         f"Normalized Mutual Information (NMI) between Louvain and Girvan-Newman: {nmi_score:.4f}\n")

            # Update the community combobox with the available community labels
            community_list = self.get_community_list()
            self.selected_community['values'] = community_list
            if community_list:
                self.selected_community.set(community_list[0])

            # Set community as node attribute for visualization
            self.node_color_attr.set('community')
            self.metrics_text.insert(END, "\nCommunity information added to node attributes for visualization\n")

        except Exception as e:
            messagebox.showerror("Error", f"Community detection comparison failed: {str(e)}")

    def get_community_list(self):
        # Collect all unique community labels from the node attributes
        communities = set()

        # Ensure that the node_attributes dictionary has the 'community' key
        for node, attrs in self.node_attributes.items():
            community = attrs.get('community')
            if community is not None:
                communities.add(str(community))  # Make sure it's a string for consistency with combobox

        # Convert the set to a sorted list (optional)
        return sorted(list(communities))

    def run_pagerank(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to analyze")
            return

        try:
            pr = nx.pagerank(g)

            # Add as node attribute
            for node in g.nodes():
                if node in self.node_attributes:
                    self.node_attributes[node]['pagerank'] = pr[node]
                else:
                    self.node_attributes[node] = {'pagerank': pr[node]}

            self.metrics_text.insert(END, "\nPageRank Results:\n")
            top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            for node, score in top_nodes:
                self.metrics_text.insert(END, f"Node {node}: {score:.4f}\n")

            self.node_size_attr.set('pagerank')
            self.metrics_text.insert(END, "PageRank scores added to node attributes for visualization\n")

        except Exception as e:
            messagebox.showerror("Error", f"PageRank calculation failed: {str(e)}")

    def run_betweenness(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to analyze")
            return

        try:
            betweenness = nx.betweenness_centrality(g)

            # Add as node attribute
            for node in g.nodes():
                if node in self.node_attributes:
                    self.node_attributes[node]['betweenness'] = betweenness[node]
                else:
                    self.node_attributes[node] = {'betweenness': betweenness[node]}

            self.metrics_text.insert(END, "\nBetweenness Centrality Results:\n")
            top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            for node, score in top_nodes:
                self.metrics_text.insert(END, f"Node {node}: {score:.4f}\n")

            self.node_color_attr.set('betweenness')
            self.metrics_text.insert(END, "Betweenness centrality added to node attributes for visualization\n")

        except Exception as e:
            messagebox.showerror("Error", f"Betweenness centrality calculation failed: {str(e)}")


if __name__ == "__main__":
    root = Tk()
    app = SocialNetworkAnalyzer(root)
    root.mainloop()
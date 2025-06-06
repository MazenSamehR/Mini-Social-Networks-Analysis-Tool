import hashlib
import os
import warnings
import webbrowser
from tkinter import *
from tkinter import filedialog, ttk, messagebox, END
import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import girvan_newman
from pyvis.network import Network
from sklearn.metrics import normalized_mutual_info_score


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
        self.main_frame = Frame(self.root, padx=10, pady=10, bg=self.current_theme["bg_main"])
        self.main_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.canvas = Canvas(self.main_frame, bg=self.current_theme["bg_main"], highlightthickness=0)
        self.scrollbar = Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill=BOTH, expand=True)

        self.control_frame = Frame(self.canvas, padx=10, pady=10, bg=self.current_theme["bg_main"])
        self.canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        self.control_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))  # Linux

        self.visualization_frame = Frame(self.root, padx=10, pady=10, bg=self.current_theme["bg_secondary"])
        self.visualization_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # UI Content
        self.title_frame = Frame(self.control_frame, bg=self.current_theme["bg_main"])
        self.title_frame.grid(row=0, column=0, columnspan=2, sticky='ew')

        Label(self.title_frame, text="Load Network Data", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).pack(side=LEFT, pady=(0, 10))

        self.theme_var = IntVar(value=0)
        self.theme_toggle = Checkbutton(self.title_frame, text="Theme", variable=self.theme_var,
                                        command=self.toggle_theme,
                                        bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                                        selectcolor=self.current_theme["button_bg"])
        self.theme_toggle.pack(side=RIGHT, pady=(0, 10), padx=(10, 0))

        # Buttons for loading data
        Button(self.control_frame, text="Load Nodes CSV", command=self.load_nodes, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=1, column=0,
                                                                                                    pady=5, sticky='ew')
        Button(self.control_frame, text="Load Edges CSV", command=self.load_edges, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=1, column=1,
                                                                                                    pady=5, sticky='ew')

        Button(self.control_frame, text="Reset Data", command=self.reset_data, width=20,
               bg=self.current_theme["error_color"], fg="white").grid(row=2, column=0, columnspan=2, pady=(10, 10),
                                                                      sticky='ew')

        # Graph Type
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
                                                           ["force-directed", "tree", "radial"], 8)
        self.layout_algo.set("force-directed")

        # Node Label Attribute
        Label(self.control_frame, text="Node Label Attribute:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=9, column=0, pady=5, sticky='w')
        self.node_label_attr = ttk.Combobox(self.control_frame, width=18)
        self.node_label_attr.grid(row=9, column=1, pady=5, sticky='ew')

        # Node Size Attribute
        Label(self.control_frame, text="Node Size Attribute:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=10, column=0, pady=5, sticky='w')
        self.node_size_attr = ttk.Combobox(self.control_frame, width=18)
        self.node_size_attr.grid(row=10, column=1, pady=5, sticky='ew')

        # Node Color Attribute
        Label(self.control_frame, text="Node Color Attribute:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=11, column=0, pady=5, sticky='w')
        self.node_color_attr = ttk.Combobox(self.control_frame, width=18)
        self.node_color_attr.grid(row=11, column=1, pady=5, sticky='ew')

        # Link Analysis
        Label(self.control_frame, text="Link Analysis", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=12, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')
        Button(self.control_frame, text="Run PageRank", command=self.run_pagerank, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=13, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')
        Button(self.control_frame, text="Run Betweenness", command=self.run_betweenness, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=14, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')

        # Community Detection
        Label(self.control_frame, text="Community Detection", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=15, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')
        self.community_algo = self.create_combobox_with_label(
            "Community Algorithm:", ["Louvain", "Girvan-Newman", "Both"], 16
        )
        self.community_algo.set("Louvain")
        Button(self.control_frame, text="Detect Communities", command=self.detect_communities, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=17, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=10,
                                                                                                    sticky='ew')

        # Filtering
        Label(self.control_frame, text="Filtering Options", font=('Arial', 16, 'bold'),
              bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"]).grid(row=18, column=0,
                                                                                          columnspan=2, pady=(20, 10),
                                                                                          sticky='w')

        self.selected_community = self.create_combobox_with_label("Filter by Community:", self.get_community_list(), 19)
        Button(self.control_frame, text="Apply Community Filter", command=self.apply_community_filter, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=20, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=5, sticky='ew')

        self.centrality_type = self.create_combobox_with_label("Filter by Centrality:",
                                                               ["degree", "betweenness", "eigenvector"], 21)

        Label(self.control_frame, text="Min Value:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=22, column=0, pady=5, sticky='w')
        self.min_centrality = Entry(self.control_frame, width=18)
        self.min_centrality.grid(row=22, column=1, pady=5, sticky='ew')

        Label(self.control_frame, text="Max Value:", bg=self.current_theme["bg_main"],
              fg=self.current_theme["text_color"]).grid(row=23, column=0, pady=5, sticky='w')
        self.max_centrality = Entry(self.control_frame, width=18)
        self.max_centrality.grid(row=23, column=1, pady=5, sticky='ew')

        Button(self.control_frame, text="Apply Centrality Filter", command=self.apply_centrality_filter, width=20,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"]).grid(row=24, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=10,
                                                                                                    sticky='ew')

        # Visualize Network
        Button(self.control_frame, text="Visualize Network", command=self.visualize_network,
               bg=self.current_theme["button_bg"], fg=self.current_theme["button_text_color"],
               font=('Arial', 12, 'bold')).grid(row=25, column=0,
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
        if self.current_theme == self.dark_theme:
            self.current_theme = self.light_theme
        else:
            self.current_theme = self.dark_theme

        self.update_ui_theme()

    def update_ui_theme(self):
        self.main_frame.config(bg=self.current_theme["bg_main"])
        self.canvas.config(bg=self.current_theme["bg_main"])
        self.scrollbar.config(bg=self.current_theme["bg_main"])
        self.control_frame.config(bg=self.current_theme["bg_main"])

        for widget in self.control_frame.winfo_children():
            if isinstance(widget, Label):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"])
            elif isinstance(widget, Button) and widget.cget("text") != "Reset Data":
                widget.config(bg=self.current_theme["button_bg"],
                              fg=self.current_theme["button_text_color"])
            elif isinstance(widget, Checkbutton):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                              selectcolor=self.current_theme["button_bg"])
            elif isinstance(widget, Radiobutton):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                              selectcolor=self.current_theme["bg_main"])

        self.visualization_frame.config(bg=self.current_theme["bg_secondary"])
        self.metrics_text.config(bg=self.current_theme["bg_secondary"], fg=self.current_theme["text_color"])
        self.theme_toggle.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"],
                                 selectcolor=self.current_theme["button_bg"])
        self.title_frame.config(bg=self.current_theme["bg_main"])

        for widget in self.title_frame.winfo_children():
            if isinstance(widget, Label):
                widget.config(bg=self.current_theme["bg_main"], fg=self.current_theme["text_color"])

    def on_graph_type_changed(self):
        self.current_graph_type = self.graph_type.get()
        self.calculate_basic_metrics()

    def reset_data(self):
        if self.original_graph is not None:
            self.graph = self.original_graph.copy()
        if self.original_directed_graph is not None:
            self.directed_graph = self.original_directed_graph.copy()

        self.node_attributes = self.original_node_attributes.copy()
        self.edge_attributes = self.original_edge_attributes.copy()

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
                    messagebox.showerror("Error",
                                         "CSV file must contain 'Source' and 'Target' columns (case insensitive)")
                    return

                if 'weight' not in df.columns:
                    df['weight'] = 1

                df = df.groupby(['source', 'target'], as_index=False).agg({'weight': 'sum'})

                if self.graph_type.get() == "undirected":
                    df[['source', 'target']] = df.apply(lambda x: sorted([x['source'], x['target']]), axis=1,
                                                        result_type='expand')
                    df = df.groupby(['source', 'target'], as_index=False).agg({'weight': 'sum'})

                self.original_graph = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True)
                self.original_directed_graph = nx.from_pandas_edgelist(df, source='source', target='target',
                                                                       edge_attr=True, create_using=nx.DiGraph())
                self.graph = self.original_graph.copy()
                self.directed_graph = self.original_directed_graph.copy()

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
            self.node_label_attr['values'] = ['id'] + list(attributes)
            if attributes:
                self.node_size_attr.set(attributes[0])
                self.node_color_attr.set(attributes[0])
                self.node_label_attr.set('id')
            else:
                self.node_size_attr['values'] = []
                self.node_color_attr['values'] = []
                self.node_label_attr['values'] = ['id']
                self.node_label_attr.set('id')

    def calculate_basic_metrics(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        self.metrics_text.delete(1.0, END)

        if g.number_of_nodes() == 0:
            self.metrics_text.insert(END, "Graph is empty - no metrics to calculate\n")
            return

        self.metrics_text.insert(END, "\n=== Network Metrics ===\n")
        self.metrics_text.insert(END, f"Number of nodes: {g.number_of_nodes()}\n")
        self.metrics_text.insert(END, f"Number of edges: {g.number_of_edges()}\n")

        if self.current_graph_type == "directed":
            try:
                if not nx.is_strongly_connected(g):
                    self.metrics_text.insert(END,
                                             "Warning: The directed graph is not strongly connected. Some metrics may not be calculable.\n")
            except nx.NetworkXPointlessConcept:
                pass

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

        try:
            self.metrics_text.insert(END, f"Average clustering coefficient: {nx.average_clustering(g):.4f}\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Could not calculate clustering coefficient: {str(e)}\n")

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
            else:
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
            # Prepare network visualization
            net = Network(
                notebook=True,
                height="100vh",
                width="100%",
                directed=(self.current_graph_type == "directed"),
                cdn_resources='remote',
                bgcolor=self.current_theme["bg_secondary"],
                font_color=self.current_theme["text_color"]
            )


            # Choose layout algorithm
            layout = self.layout_algo.get()

            # Auto-calculate spring constant based on graph size
            spring_constant = self.calculate_optimal_spring_constant(g)

            # Auto-select central node based on highest degree centrality
            central_node = self.find_optimal_central_node(g)

            # Generate positions based on layout algorithm
            positions = self.generate_layout(g, layout, spring_constant, central_node)

            # Scale positions to fit in visualization
            positions = self.scale_positions(positions)

            # Add nodes with positions
            for node in g.nodes():
                label_attr = self.node_label_attr.get()
                node_label = str(node) if label_attr == 'id' else str(
                    self.node_attributes.get(node, {}).get(label_attr, node))

                node_attrs = {
                    "id": str(node),
                    "label": node_label,
                    "title": str(node),
                    "borderWidth": 1,
                    "shadow": True
                }

                # Apply layout positions
                if positions and node in positions:
                    node_attrs["x"] = positions[node][0]
                    node_attrs["y"] = positions[node][1]
                    node_attrs["physics"] = False  # Disable physics for pre-positioned nodes

                if self.node_attributes and node in self.node_attributes:
                    for attr, value in self.node_attributes[node].items():
                        node_attrs[attr] = value
                        if attr == self.node_size_attr.get():
                            node_attrs["size"] = float(value) * 2 if str(value).replace('.', '').isdigit() else 10
                        if attr == self.node_color_attr.get():
                            node_attrs["color"] = self.value_to_color(value)
                        if attr == "shape":
                            node_attrs["shape"] = value if value in ['dot', 'square', 'triangle', 'star'] else 'dot'

                selected_shape = self.node_shape_attr.get()
                node_attrs["shape"] = selected_shape if selected_shape in ['dot', 'square', 'triangle',
                                                                           'star'] else 'dot'

                net.add_node(node_attrs["id"], **node_attrs)

            # Add edges with styling
            added_edges = set()
            edge_weight_threshold = 0  # Set to 0 to show all edges

            for edge in g.edges():
                u, v = map(str, edge)

                if self.current_graph_type == "undirected":
                    edge_key = frozenset({u, v})
                    if edge_key in added_edges:
                        continue
                    added_edges.add(edge_key)
                    u, v = sorted((u, v))

                key = (u, v) if (u, v) in self.edge_attributes else (v, u)

                edge_attrs = {}
                if key in self.edge_attributes:
                    for attr, value in self.edge_attributes[key].items():
                        if attr == "weight":
                            if float(value) < edge_weight_threshold:
                                continue
                            edge_attrs["width"] = float(value) / 2 if value else 1
                            edge_attrs["title"] = f"Weight: {value}"
                        elif attr == "color":
                            edge_attrs["color"] = self.value_to_color(value)

                selected_style = self.edge_style_attr.get()
                if selected_style == "dashed":
                    edge_attrs["dashes"] = True
                elif selected_style == "dotted":
                    edge_attrs["dashes"] = [2, 5]

                # Special styling for tree layout
                if layout == "tree" or layout == "radial":
                    edge_attrs["smooth"] = {"type": "curvedCW", "roundness": 0.2}
                    if self.current_graph_type == "directed":
                        edge_attrs["arrows"] = "to"

                # Default edge styling
                if self.current_graph_type == "directed" and "arrows" not in edge_attrs:
                    edge_attrs["arrows"] = "to"
                    edge_attrs["smooth"] = {"type": "curvedCW", "roundness": 0.2}

                net.add_edge(u, v, **edge_attrs)

            # Configure options for different layouts
            options = {
                "interaction": {"hover": True, "navigationButtons": True},
                "physics": {
                    "enabled": positions is None,  # Only enable physics if no positions
                    "barnesHut": {"gravitationalConstant": -2000, "centralGravity": 0.3,
                                  "springLength": spring_constant, "springConstant": 0.04}
                }
            }

            # Add specific options for different layouts
            if layout == "force-directed":
                options["physics"]["solver"] = "barnesHut"
            elif layout == "tree":
                options["layout"] = {"hierarchical": {"enabled": False}}  # We're using our own layout
                options["physics"]["enabled"] = False
            elif layout == "radial":
                options["physics"]["enabled"] = False

            net.options = options

            # Output and open in browser
            output_file = "network_visualization.html"
            net.save_graph(output_file)
            #custom CSS for the navigation buttons
            with open(output_file, 'r+') as f:
                content = f.read()

                # custom CSS
                custom_css = """
                        <style>                           

                            /* All navigation buttons */
                            .vis-navigation .vis-button {
                                background-color: #BB86FC !important;
                            }
                          
                        </style>
                        """
                content = content.replace('</head>', custom_css + '</head>')

                # Write back to the file
                f.seek(0)
                f.write(content)
                f.truncate()

            webbrowser.open('file://' + os.path.realpath(output_file))

        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_optimal_spring_constant(self, g):
        """Calculate optimal spring constant based on graph properties"""
        if len(g.nodes()) == 0:
            return 100  # Default value

        # Base spring constant on graph density and size
        density = nx.density(g)
        num_nodes = len(g.nodes())

        # For very dense graphs, use smaller spring constant
        if density > 0.1:
            return 50 + (num_nodes * 0.5)
        # For medium density
        elif density > 0.01:
            return 75 + (num_nodes * 0.75)
        # For sparse graphs
        else:
            return 100 + (num_nodes * 1.0)

    def find_optimal_central_node(self, g):
        """Find the best central node based on graph properties"""
        if len(g.nodes()) == 0:
            return None

        # Try betweenness centrality first (most meaningful for centrality)
        try:
            betweenness = nx.betweenness_centrality(g)
            max_node = max(betweenness, key=betweenness.get)
            if betweenness[max_node] > 0:  # Only use if meaningful centrality exists
                return max_node
        except:
            pass

        # Fallback to degree centrality
        try:
            degrees = dict(g.degree())
            max_degree = max(degrees.values())
            if max_degree > 0:  # Only use if non-isolated nodes exist
                candidates = [n for n, d in degrees.items() if d == max_degree]
                return candidates[0]  # Return first node with max degree
        except:
            pass

        # Final fallbacks
        if nx.is_directed(g):
            # For directed graphs, try to find a node with high in-degree
            try:
                in_degrees = dict(g.in_degree())
                return max(in_degrees, key=in_degrees.get)
            except:
                pass

        # Return first node as last resort
        return next(iter(g.nodes())) if g.nodes() else None

    def generate_layout(self, g, layout_name, spring_constant, central_node=None):
        """Generate node positions for different layout types.

        Args:
            g: NetworkX graph
            layout_name: One of "force-directed", "tree", or "radial"
            spring_constant: Parameter for force-directed layout
            central_node: Optional central node for tree/radial layouts

        Returns:
            Dictionary of node positions {node: (x, y)}
        """
        if not g.nodes():
            return None

        if not central_node:
            central_node = self.find_optimal_central_node(g)

        # Initialize positions dictionary
        pos = {}

        if layout_name == "force-directed":
            # Spring layout with optimized parameters
            return nx.spring_layout(
                g,
                k=spring_constant / 100,  # Optimal spacing
                iterations=100,
                seed=42  # For reproducibility
            )

        elif layout_name == "tree":
            # Hierarchical tree layout (top-down)

            # Assign levels using DFS (ensures parent-child relationships)
            levels = {central_node: 0}
            stack = [(central_node, 0)]  # (node, level)

            while stack:
                node, level = stack.pop()
                for neighbor in g.neighbors(node):
                    if neighbor not in levels:
                        levels[neighbor] = level + 1
                        stack.append((neighbor, level + 1))

            # Group nodes by level
            layers = {}
            for node, level in levels.items():
                if level not in layers:
                    layers[level] = []
                layers[level].append(node)

            # Position nodes (root at top, children below)
            max_level = max(layers.keys()) if layers else 1
            y_spacing = 1.0 / (max_level + 1)

            for level, nodes in layers.items():
                # Evenly space nodes horizontally
                x_positions = np.linspace(0, 1, len(nodes))
                for i, node in enumerate(nodes):
                    pos[node] = (
                        x_positions[i],  # x position (evenly spaced)
                        level * y_spacing  # y increases downward
                    )

        elif layout_name == "radial":
            # Radial layout with concentric circles

            # Assign levels using BFS (distance from center)
            levels = {central_node: 0}
            queue = [central_node]

            while queue:
                node = queue.pop(0)
                for neighbor in g.neighbors(node):
                    if neighbor not in levels:
                        levels[neighbor] = levels[node] + 1
                        queue.append(neighbor)

            # Group nodes by level (shell)
            shells = {}
            for node, level in levels.items():
                if level not in shells:
                    shells[level] = []
                shells[level].append(node)

            # Position nodes in concentric circles
            max_level = max(shells.keys()) if shells else 1
            radius_step = 1.0 / (max_level + 1)

            for level, nodes in shells.items():
                radius = level * radius_step
                angle_step = 2 * np.pi / len(nodes) if nodes else 0

                for i, node in enumerate(nodes):
                    angle = i * angle_step
                    pos[node] = (
                        radius * np.cos(angle),  # x position
                        radius * np.sin(angle)  # y position
                    )

        else:
            # Fallback to spring layout
            return nx.spring_layout(g)

        # Scale positions to [-1, 1] range
        if pos:
            x_vals = [p[0] for p in pos.values()]
            y_vals = [p[1] for p in pos.values()]

            if x_vals and y_vals:
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)

                x_range = x_max - x_min if x_max != x_min else 1
                y_range = y_max - y_min if y_max != y_min else 1

                # Normalize to [-1, 1] range
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (
                        2 * ((x - x_min) / x_range - 0.5),  # x in [-1, 1]
                        2 * ((y - y_min) / y_range - 0.5)  # y in [-1, 1]
                    )

        return pos

    def scale_positions(self, positions, scale=500):
        """Scale positions to fit within visualization bounds"""
        if not positions:
            return None

        # Find the range of positions
        x_vals = [pos[0] for pos in positions.values()]
        y_vals = [pos[1] for pos in positions.values()]

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # Calculate scaling factors
        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range == 0:
            x_range = 1
        if y_range == 0:
            y_range = 1

        # Scale and center positions
        scaled_positions = {}
        for node, pos in positions.items():
            x_scaled = ((pos[0] - x_min) / x_range - 0.5) * scale
            y_scaled = ((pos[1] - y_min) / y_range - 0.5) * scale
            scaled_positions[node] = (x_scaled, y_scaled)

        return scaled_positions

    def value_to_color(self, value):
        if value is None:
            return "hsl(0, 0%, 50%)"

        if isinstance(value, (int, float)):
            value = f"{float(value):.4f}"
        else:
            value = str(value)

        hash_val = int(hashlib.sha256(value.encode()).hexdigest(), 16)

        h = hash_val % 360
        s = 80 + (hash_val % 21)
        l = 40 + (hash_val % 21)

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

        try:
            selected_community = int(selected_community)
        except ValueError:
            pass

        try:

            filtered_nodes = [n for n in g.nodes() if
                              self.node_attributes.get(n, {}).get('community') == selected_community]

            if not filtered_nodes:
                messagebox.showwarning("Warning", "No nodes match the selected community")
                return

            filtered_graph = g.subgraph(filtered_nodes)

            if self.current_graph_type == "undirected":
                self.graph = filtered_graph
            else:
                self.directed_graph = filtered_graph

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

        algo = self.community_algo.get()

        try:

            louvain_partition = None
            louvain_modularity = None
            louvain_communities = None
            girvan_communities = None
            girvan_edges_removed = None

            if algo == "Louvain" or algo == "Both":
                louvain_partition = community_louvain.best_partition(g)
                louvain_modularity = community_louvain.modularity(louvain_partition, g)
                louvain_communities = {comm_id: [] for comm_id in louvain_partition.values()}
                for node, comm_id in louvain_partition.items():
                    louvain_communities[comm_id].append(node)

            if algo == "Girvan-Newman" or algo == "Both":
                comp = girvan_newman(g)
                girvan_communities = tuple(sorted(c) for c in next(comp))
                original_edges = len(g.edges())
                remaining_edges = sum(g.subgraph(comm).number_of_edges() for comm in girvan_communities)
                girvan_edges_removed = original_edges - remaining_edges

            self.metrics_text.insert(END, "\nCommunity Detection Comparison:\n")

            def get_community_sizes(community_dict):
                return [len(nodes) for nodes in community_dict.values()]

            def conductance(graph, community_nodes):
                cut_size = sum(1 for u in community_nodes for v in graph.neighbors(u) if v not in community_nodes)
                volume = len(community_nodes) * (len(graph.nodes()) - len(community_nodes))
                return cut_size / volume if volume > 0 else 0

            if louvain_partition:
                community_sizes = get_community_sizes(louvain_communities)
                self.metrics_text.insert(END, f"\nLouvain Algorithm Results:\n")
                self.metrics_text.insert(END, f"Number of communities detected: {len(louvain_communities)}\n")
                self.metrics_text.insert(END, f"Community Size Distribution: {community_sizes}\n")

                for comm_id, nodes in louvain_communities.items():
                    community_conductance = conductance(g, nodes)
                    self.metrics_text.insert(END, f"Conductance of community {comm_id}: {community_conductance:.4f}\n")
                    self.metrics_text.insert(END, f"Nodes in community {comm_id}: {', '.join(map(str, nodes))}\n")

                for node in g.nodes():
                    if node in self.node_attributes:
                        self.node_attributes[node]['community'] = louvain_partition[node]
                    else:
                        self.node_attributes[node] = {'community': louvain_partition[node]}

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
                        if node in self.node_attributes:
                            self.node_attributes[node]['community'] = i
                        else:
                            self.node_attributes[node] = {'community': i}

            self.metrics_text.insert(END, "\nClustering Evaluation:\n")

            if louvain_modularity:
                self.metrics_text.insert(END, f"Louvain Modularity: {louvain_modularity:.4f}\n")

            if girvan_communities:
                girvan_partition = {node: next(i for i, c in enumerate(girvan_communities) if node in c) for node in
                                    g.nodes()}
                girvan_modularity = community_louvain.modularity(girvan_partition, g)
                self.metrics_text.insert(END, f"Girvan-Newman Modularity: {girvan_modularity:.4f}\n")

            if louvain_partition and girvan_communities:
                louvain_labels = [louvain_partition[node] for node in g.nodes()]
                girvan_labels = [next(i for i, c in enumerate(girvan_communities) if node in c) for node in g.nodes()]
                nmi_score = normalized_mutual_info_score(louvain_labels, girvan_labels)
                self.metrics_text.insert(END,
                                         f"Normalized Mutual Information (NMI) between Louvain and Girvan-Newman: {nmi_score:.4f}\n")

            community_list = self.get_community_list()
            self.selected_community['values'] = community_list
            if community_list:
                self.selected_community.set(community_list[0])

            self.node_color_attr.set('community')
            self.metrics_text.insert(END, "\nCommunity information added to node attributes for visualization\n")

        except Exception as e:
            messagebox.showerror("Error", f"Community detection comparison failed: {str(e)}")

    def get_community_list(self):

        communities = set()

        for node, attrs in self.node_attributes.items():
            community = attrs.get('community')
            if community is not None:
                communities.add(str(community))

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
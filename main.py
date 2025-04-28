import os
import warnings
import webbrowser
from tkinter import *
from tkinter import filedialog, ttk, messagebox, END
import community as community_louvain
import networkx as nx
import pandas as pd
from community import community_louvain
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

        self.setup_ui()

    def setup_ui(self):
        # Main frame with scrollable content
        main_frame = Frame(self.root, padx=20, pady=20, bg="#f5f5f5")
        main_frame.pack(side=LEFT, fill=Y, expand=True)

        # Scrollbar and canvas for scrolling
        canvas = Canvas(main_frame, bg="#f5f5f5", highlightthickness=0)
        scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create a frame inside the canvas that will hold all the widgets
        control_frame = Frame(canvas, padx=20, pady=20, bg="#f5f5f5")
        canvas.create_window((0, 0), window=control_frame, anchor="nw")

        control_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        visualization_frame = Frame(self.root, padx=10, pady=10, bg="white")
        visualization_frame.pack(side=RIGHT, expand=True, fill=BOTH)

        # Load data section
        Label(control_frame, text="Load Network Data", font=('Arial', 16, 'bold'), bg="#f5f5f5").grid(row=0, column=0,
                                                                                                      columnspan=2,
                                                                                                      pady=(0, 15),
                                                                                                      sticky='w')
        Button(control_frame, text="Load Nodes CSV", command=self.load_nodes, width=20).grid(row=1, column=0, pady=5,
                                                                                             sticky='ew')
        Button(control_frame, text="Load Edges CSV", command=self.load_edges, width=20).grid(row=1, column=1, pady=5,
                                                                                             sticky='ew')

        # Reset button
        Button(control_frame, text="Reset Data", command=self.reset_data, width=20, bg="#ff4d4d", fg="white").grid(
            row=2, column=0, columnspan=2, pady=(10, 20), sticky='ew')

        # Graph type selection
        Label(control_frame, text="Graph Type:", font=('Arial', 14), bg="#f5f5f5").grid(row=3, column=0, pady=5,
                                                                                        sticky='w')
        self.graph_type = StringVar(value="undirected")
        Radiobutton(control_frame, text="Undirected", variable=self.graph_type, value="undirected",
                    command=self.on_graph_type_changed, bg="#f5f5f5").grid(row=4, column=0, sticky='w')
        Radiobutton(control_frame, text="Directed", variable=self.graph_type, value="directed",
                    command=self.on_graph_type_changed, bg="#f5f5f5").grid(row=4, column=1, sticky='w')

        # Visualization options
        Label(control_frame, text="Visualization Options", font=('Arial', 16, 'bold'), bg="#f5f5f5").grid(row=5,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          pady=(20, 10),
                                                                                                          sticky='w')

        Label(control_frame, text="Node Shape Attribute:", bg="#f5f5f5").grid(row=6, column=0, pady=5, sticky='w')
        self.node_shape_attr = ttk.Combobox(control_frame, values=["dot", "square", "triangle", "star"], width=18)
        self.node_shape_attr.grid(row=6, column=1, pady=5, sticky='ew')
        self.node_shape_attr.set("dot")

        Label(control_frame, text="Edge Style:", bg="#f5f5f5").grid(row=7, column=0, pady=5, sticky='w')
        self.edge_style_attr = ttk.Combobox(control_frame, values=["solid", "dashed", "dotted"], width=18)
        self.edge_style_attr.grid(row=7, column=1, pady=5, sticky='ew')
        self.edge_style_attr.set("solid")

        Label(control_frame, text="Layout Algorithm:", bg="#f5f5f5").grid(row=8, column=0, pady=5, sticky='w')
        self.layout_algo = ttk.Combobox(control_frame, values=["force-directed", "hierarchical", "circular", "random"],
                                        width=18)
        self.layout_algo.grid(row=8, column=1, pady=5, sticky='ew')
        self.layout_algo.set("force-directed")

        Label(control_frame, text="Node Size Attribute:", bg="#f5f5f5").grid(row=9, column=0, pady=5, sticky='w')
        self.node_size_attr = ttk.Combobox(control_frame, width=18)
        self.node_size_attr.grid(row=9, column=1, pady=5, sticky='ew')

        Label(control_frame, text="Node Color Attribute:", bg="#f5f5f5").grid(row=10, column=0, pady=5, sticky='w')
        self.node_color_attr = ttk.Combobox(control_frame, width=18)
        self.node_color_attr.grid(row=10, column=1, pady=5, sticky='ew')

        # Link Analysis section
        Label(control_frame, text="Link Analysis", font=('Arial', 16, 'bold'), bg="#f5f5f5").grid(row=11, column=0,
                                                                                                  columnspan=2,
                                                                                                  pady=(20, 10),
                                                                                                  sticky='w')
        Button(control_frame, text="Run PageRank", command=self.run_pagerank, width=20).grid(row=12, column=0,
                                                                                             columnspan=2, pady=5,
                                                                                             sticky='ew')
        Button(control_frame, text="Run Betweenness", command=self.run_betweenness, width=20).grid(row=13, column=0,
                                                                                                   columnspan=2, pady=5,
                                                                                                   sticky='ew')

        # Community Detection section
        Label(control_frame, text="Community Detection", font=('Arial', 16, 'bold'), bg="#f5f5f5").grid(row=14,
                                                                                                        column=0,
                                                                                                        columnspan=2,
                                                                                                        pady=(20, 10),
                                                                                                        sticky='w')
        self.community_algo = ttk.Combobox(control_frame, values=["Louvain", "Girvan-Newman", "Both"], width=18)
        self.community_algo.grid(row=15, column=0, columnspan=2, pady=5, sticky='ew')
        Button(control_frame, text="Detect Communities", command=self.detect_communities, width=20).grid(row=16,
                                                                                                         column=0,
                                                                                                         columnspan=2,
                                                                                                         pady=10,
                                                                                                         sticky='ew')

        # Filtering options
        Label(control_frame, text="Filtering Options", font=('Arial', 16, 'bold'), bg="#f5f5f5").grid(row=17, column=0,
                                                                                                      columnspan=2,
                                                                                                      pady=(20, 10),
                                                                                                      sticky='w')

        Label(control_frame, text="Filter by Community:", bg="#f5f5f5").grid(row=18, column=0, pady=5, sticky='w')
        self.selected_community = ttk.Combobox(control_frame, values=self.get_community_list(), width=18)
        self.selected_community.grid(row=18, column=1, pady=5, sticky='ew')
        Button(control_frame, text="Apply Community Filter", command=self.apply_community_filter, width=20).grid(row=19,
                                                                                                                 column=0,
                                                                                                                 columnspan=2,
                                                                                                                 pady=5,
                                                                                                                 sticky='ew')

        Label(control_frame, text="Filter by Centrality:", bg="#f5f5f5").grid(row=20, column=0, pady=5, sticky='w')
        self.centrality_type = ttk.Combobox(control_frame, values=["degree", "betweenness", "eigenvector"], width=18)
        self.centrality_type.grid(row=20, column=1, pady=5, sticky='ew')

        Label(control_frame, text="Min Value:", bg="#f5f5f5").grid(row=21, column=0, pady=5, sticky='w')
        self.min_centrality = Entry(control_frame, width=18)
        self.min_centrality.grid(row=21, column=1, pady=5, sticky='ew')

        Label(control_frame, text="Max Value:", bg="#f5f5f5").grid(row=22, column=0, pady=5, sticky='w')
        self.max_centrality = Entry(control_frame, width=18)
        self.max_centrality.grid(row=22, column=1, pady=5, sticky='ew')

        Button(control_frame, text="Apply Centrality Filter", command=self.apply_centrality_filter, width=20).grid(
            row=23, column=0, columnspan=2, pady=10, sticky='ew')

        # Visualize button
        Button(control_frame, text="Visualize Network", command=self.visualize_network, bg="#4CAF50", fg="white",
               font=('Arial', 12, 'bold')).grid(row=24, column=0, columnspan=2, pady=20, sticky='ew')

        # Metrics display
        self.metrics_text = Text(visualization_frame, width=80, height=30, bg="#f0f0f0", font=('Arial', 10))
        self.metrics_text.pack(expand=True, fill=BOTH)

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

        # Check for strong connectivity in directed graphs and issue warning
        if self.current_graph_type == "directed":
            try:
                if not nx.is_strongly_connected(g):
                    self.metrics_text.insert(END, "Warning: The directed graph is not strongly connected. Some metrics may not be calculable.\n")
            except nx.NetworkXPointlessConcept:
                pass  # Graph is empty, handled elsewhere

        # Average Shortest Path Length
        try:
            if self.current_graph_type == "directed":
                if nx.is_strongly_connected(g):
                    avg_path = nx.average_shortest_path_length(g)
                    self.metrics_text.insert(END, f"Average shortest path length (directed): {avg_path:.4f}\n")
                else:
                    self.metrics_text.insert(END, "Cannot calculate average shortest path length: directed graph is not strongly connected.\n")
            else:
                if nx.is_connected(g):
                    avg_path = nx.average_shortest_path_length(g)
                    self.metrics_text.insert(END, f"Average shortest path length: {avg_path:.4f}\n")
                else:
                    self.metrics_text.insert(END, "Cannot calculate average shortest path length: graph is not connected.\n")
        except nx.NetworkXPointlessConcept:
            self.metrics_text.insert(END, "Graph is empty - cannot calculate average shortest path length.\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Error calculating average shortest path length: {str(e)}\n")

        # Average Clustering Coefficient
        try:
            self.metrics_text.insert(END, f"Average clustering coefficient: {nx.average_clustering(g):.4f}\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Could not calculate clustering coefficient: {str(e)}\n")

        # Degree Assortativity Coefficient
        try:
            self.metrics_text.insert(END,
                                     f"Degree assortativity coefficient: {nx.degree_assortativity_coefficient(g):.4f}\n")
        except nx.NetworkXError as e:
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

        # Eigenvector Centrality
        try:
            eigenvector_centrality = nx.eigenvector_centrality(g)
            self.metrics_text.insert(END, f"Eigenvector Centrality: {eigenvector_centrality}\n")
        except Exception as e:
            self.metrics_text.insert(END, f"Could not calculate eigenvector centrality: {str(e)}\n")

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

                if self.edge_attributes and edge in self.edge_attributes:
                    for attr, value in self.edge_attributes[edge].items():
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
                net.hierarchical_layout()

            # Save and show
            output_file = "network_visualization.html"
            net.show(output_file)
            webbrowser.open('file://' + os.path.realpath(output_file))

        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")

    def value_to_color(self, value):
        import hashlib

        # If value is numeric (like weight), don't handle it here
        if isinstance(value, (int, float)):
            value = str(value)

        # Use a stable hash to generate different colors
        hash_val = int(hashlib.md5(value.encode()).hexdigest(), 16)
        # Map to distinct HSL colors
        h = hash_val % 360  # Hue: full circle
        s = 90 + (hash_val % 10)  # Saturation: between 90-100% for bright colors
        l = 50  # Lightness: fixed at 50% for contrast

        return f"hsl({h}, {s}%, {l}%)"

    def apply_centrality_filter(self):
        if self.current_graph_type == "undirected":
            g = self.graph
        else:
            g = self.directed_graph

        if g.number_of_nodes() == 0:
            messagebox.showwarning("Warning", "The graph is empty - nothing to filter")
            return

        # Get filter options from the user
        centrality_type = self.centrality_type.get()
        min_val = float(self.min_centrality.get()) if self.min_centrality.get() else 0
        max_val = float(self.max_centrality.get()) if self.max_centrality.get() else float('inf')

        try:
            # Apply centrality filtering
            if centrality_type == "degree":
                centrality = nx.degree_centrality(g)
            elif centrality_type == "betweenness":
                centrality = nx.betweenness_centrality(g)
            elif centrality_type == "eigenvector":
                try:
                    centrality = nx.eigenvector_centrality(g)
                except nx.NetworkXError:
                    messagebox.showwarning("Warning",
                                           "Eigenvector centrality cannot be computed for this graph (may be disconnected)")
                    return
            else:
                messagebox.showwarning("Warning", "Please select a centrality type")
                return

            # Filter nodes based on centrality score range
            filtered_nodes = [n for n, c in centrality.items() if min_val <= c <= max_val]

            if not filtered_nodes:
                messagebox.showwarning("Warning", "No nodes match the filter criteria")
                return

            # Create subgraph with filtered nodes
            filtered_graph = g.subgraph(filtered_nodes)

            if self.current_graph_type == "undirected":
                self.graph = filtered_graph
            else:
                self.directed_graph = filtered_graph

            # Display filtering results
            self.metrics_text.insert(END,
                                     f"\nApplied {centrality_type} centrality filter: {len(filtered_nodes)}/{g.number_of_nodes()} nodes remain\n")
            self.calculate_basic_metrics()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during centrality filtering: {str(e)}")

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
                girvan_edges_removed = len(g.edges()) - len(girvan_communities)  # Number of edges removed

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
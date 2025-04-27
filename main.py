import networkx as nx
import pandas as pd
from pyvis.network import Network
import community as community_louvain
from networkx.algorithms.community import girvan_newman
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, ttk, messagebox
import webbrowser
import os
import warnings


class SocialNetworkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Social Networks Analysis Tool")
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.current_graph_type = "undirected"
        self.node_attributes = {}
        self.edge_attributes = {}

        # Suppress PyVis warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        self.setup_ui()

    def setup_ui(self):
        # Main frames
        control_frame = Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=LEFT, fill=Y)

        visualization_frame = Frame(self.root, padx=10, pady=10)
        visualization_frame.pack(side=RIGHT, expand=True, fill=BOTH)

        # Load data section
        Label(control_frame, text="Load Network Data", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2,
                                                                                        pady=5)
        Button(control_frame, text="Load Nodes CSV", command=self.load_nodes).grid(row=1, column=0, pady=5)
        Button(control_frame, text="Load Edges CSV", command=self.load_edges).grid(row=1, column=1, pady=5)

        # Graph type
        Label(control_frame, text="Graph Type:", font=('Arial', 10)).grid(row=2, column=0, pady=5)
        self.graph_type = StringVar(value="undirected")
        Radiobutton(control_frame, text="Undirected", variable=self.graph_type, value="undirected").grid(row=3,
                                                                                                         column=0)
        Radiobutton(control_frame, text="Directed", variable=self.graph_type, value="directed").grid(row=3, column=1)

        # Visualization options
        Label(control_frame, text="Visualization Options", font=('Arial', 12, 'bold')).grid(row=4, column=0,
                                                                                            columnspan=2, pady=10)

        # Node Shape Selection
        Label(control_frame, text="Node Shape Attribute:").grid(row=5, column=0, pady=5)
        self.node_shape_attr = ttk.Combobox(control_frame, values=["dot", "square", "triangle", "star"])
        self.node_shape_attr.grid(row=5, column=1, pady=5)
        self.node_shape_attr.set("dot")

        # Edge Style Selection
        Label(control_frame, text="Edge Style:").grid(row=6, column=0, pady=5)
        self.edge_style_attr = ttk.Combobox(control_frame, values=["solid", "dashed", "dotted"])
        self.edge_style_attr.grid(row=6, column=1, pady=5)
        self.edge_style_attr.set("solid")

        # Layout algorithms
        Label(control_frame, text="Layout Algorithm:").grid(row=7, column=0, pady=5)
        self.layout_algo = ttk.Combobox(control_frame, values=["force-directed", "hierarchical", "circular", "random"])
        self.layout_algo.grid(row=7, column=1, pady=5)
        self.layout_algo.set("force-directed")

        # Node attributes
        Label(control_frame, text="Node Size Attribute:").grid(row=8, column=0, pady=5)
        self.node_size_attr = ttk.Combobox(control_frame)
        self.node_size_attr.grid(row=8, column=1, pady=5)

        Label(control_frame, text="Node Color Attribute:").grid(row=9, column=0, pady=5)
        self.node_color_attr = ttk.Combobox(control_frame)
        self.node_color_attr.grid(row=9, column=1, pady=5)

        # Filtering options
        Label(control_frame, text="Filtering Options", font=('Arial', 12, 'bold')).grid(row=10, column=0, columnspan=2,
                                                                                        pady=10)

        Label(control_frame, text="Filter by Centrality:").grid(row=11, column=0, pady=5)
        self.centrality_type = ttk.Combobox(control_frame, values=["degree", "betweenness", "eigenvector"])
        self.centrality_type.grid(row=11, column=1, pady=5)

        Label(control_frame, text="Min Value:").grid(row=12, column=0, pady=5)
        self.min_centrality = Entry(control_frame)
        self.min_centrality.grid(row=12, column=1, pady=5)

        Label(control_frame, text="Max Value:").grid(row=13, column=0, pady=5)
        self.max_centrality = Entry(control_frame)
        self.max_centrality.grid(row=13, column=1, pady=5)

        Button(control_frame, text="Apply Filter", command=self.apply_filter).grid(row=14, column=0, columnspan=2,
                                                                                   pady=10)

        # Community detection
        Label(control_frame, text="Community Detection", font=('Arial', 12, 'bold')).grid(row=15, column=0,
                                                                                          columnspan=2, pady=10)

        self.community_algo = ttk.Combobox(control_frame, values=["Louvain", "Girvan-Newman"])
        self.community_algo.grid(row=16, column=0, columnspan=2, pady=5)

        Button(control_frame, text="Detect Communities", command=self.detect_communities).grid(row=17, column=0,
                                                                                               columnspan=2, pady=10)

        # Link analysis
        Label(control_frame, text="Link Analysis", font=('Arial', 12, 'bold')).grid(row=18, column=0, columnspan=2,
                                                                                    pady=10)

        Button(control_frame, text="Run PageRank", command=self.run_pagerank).grid(row=19, column=0, columnspan=2,
                                                                                   pady=5)
        Button(control_frame, text="Run Betweenness", command=self.run_betweenness).grid(row=20, column=0, columnspan=2,
                                                                                         pady=5)

        # Visualize button
        Button(control_frame, text="Visualize Network", command=self.visualize_network,
               bg="lightblue", font=('Arial', 10, 'bold')).grid(row=21, column=0, columnspan=2, pady=20)

        # Metrics display
        self.metrics_text = Text(visualization_frame, width=80, height=30)
        self.metrics_text.pack(expand=True, fill=BOTH)

    def load_nodes(self):
        filepath = filedialog.askopenfilename(title="Select Nodes CSV File", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                df = pd.read_csv(filepath)

                df.columns = [col.lower() for col in df.columns]

                if 'id' not in df.columns:
                    messagebox.showerror("Error", "CSV file must contain an 'id' column (case insensitive)")
                    return

                self.node_attributes = df.set_index('id').to_dict('index')
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

                if self.graph_type.get() == "undirected":
                    self.graph = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True)
                    self.current_graph_type = "undirected"
                else:
                    self.directed_graph = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True,
                                                                  create_using=nx.DiGraph())
                    self.current_graph_type = "directed"

                # Edge attributes
                self.edge_attributes = df.set_index(['source', 'target']).to_dict('index')
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

        try:
            if nx.is_connected(g.to_undirected() if self.current_graph_type == "directed" else g):
                self.metrics_text.insert(END,
                                         f"Average shortest path length: {nx.average_shortest_path_length(g):.4f}\n")
            else:
                self.metrics_text.insert(END,
                                         "Graph is not connected - cannot calculate average shortest path length\n")
        except nx.NetworkXPointlessConcept:
            self.metrics_text.insert(END, "Graph is empty - cannot calculate connectivity\n")

        try:
            self.metrics_text.insert(END, f"Average clustering coefficient: {nx.average_clustering(g):.4f}\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Could not calculate clustering coefficient: {str(e)}\n")

        try:
            self.metrics_text.insert(END,
                                     f"Degree assortativity coefficient: {nx.degree_assortativity_coefficient(g):.4f}\n")
        except nx.NetworkXError as e:
            self.metrics_text.insert(END, f"Could not calculate assortativity: {str(e)}\n")

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

            # Configure arrow display for directed graphs
            if self.current_graph_type == "directed":
                net.set_options("""
                {
                  "edges": {
                    "arrows": {
                      "to": {
                        "enabled": true,
                        "scaleFactor": 1.5
                      }
                    },
                    "smooth": false
                  }
                }
                """)

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
                        if attr == "weight":
                            edge_attrs["width"] = float(value) / 10 if value else 1
                        if attr == "color":
                            edge_attrs["color"] = self.value_to_color(value)
                        if attr == "style":
                            edge_attrs["dashes"] = True if value == "dashed" else False
                        if attr == "width":
                            edge_attrs["width"] = float(value)

                net.add_edge(edge[0], edge[1], **edge_attrs)

            # Apply layout
            layout = self.layout_algo.get()
            if layout == "force-directed":
                net.force_atlas_2based()
            elif layout == "hierarchical":
                net.hrepulsion()

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

    def apply_filter(self):
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
                try:
                    centrality = nx.eigenvector_centrality(g)
                except nx.NetworkXError:
                    messagebox.showwarning("Warning",
                                           "Eigenvector centrality cannot be computed for this graph (may be disconnected)")
                    return
            else:
                messagebox.showwarning("Warning", "Please select a centrality type")
                return

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

            self.metrics_text.insert(END,
                                     f"\nApplied {centrality_type} centrality filter: {len(filtered_nodes)}/{g.number_of_nodes()} nodes remain\n")
            self.calculate_basic_metrics()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during filtering: {str(e)}")

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
            if algo == "Louvain":
                partition = community_louvain.best_partition(g)
                communities = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)

                modularity = community_louvain.modularity(partition, g)
                self.metrics_text.insert(END, f"\nLouvain Community Detection:\n")
                self.metrics_text.insert(END, f"Number of communities: {len(communities)}\n")
                self.metrics_text.insert(END, f"Modularity: {modularity:.4f}\n")

                # Add community as node attribute for visualization
                for node in g.nodes():
                    if node in self.node_attributes:
                        self.node_attributes[node]['community'] = partition[node]
                    else:
                        self.node_attributes[node] = {'community': partition[node]}

            elif algo == "Girvan-Newman":
                comp = girvan_newman(g)
                communities = tuple(sorted(c) for c in next(comp))
                self.metrics_text.insert(END, f"\nGirvan-Newman Community Detection:\n")
                self.metrics_text.insert(END, f"Number of communities: {len(communities)}\n")

                # For visualization, we'll just use the first level communities
                for i, comm in enumerate(communities):
                    for node in comm:
                        if node in self.node_attributes:
                            self.node_attributes[node]['community'] = i
                        else:
                            self.node_attributes[node] = {'community': i}

            self.node_color_attr.set('community')
            self.metrics_text.insert(END, "Community information added to node attributes for visualization\n")

        except Exception as e:
            messagebox.showerror("Error", f"Community detection failed: {str(e)}")

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
            top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
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
            top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
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
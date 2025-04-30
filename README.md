# Social Network Analysis Tool

An interactive tool for visualizing and analyzing social networks using Python, NetworkX, and Pyvis.

---

## 🔧 Features

### 🟢 Node & Edge Customization
- Node: size, color, shape, label (from CSV)
- Edge: color, solid/dashed/dotted styles

### 🌐 Visualization
- Built using **Pyvis**
- Supports **directed** and **undirected** graphs

### 🧭 Layout Algorithms
- Force-Directed
- Tree (Hierarchical)
- Radial

### 📊 Graph Metrics
- Degree Distribution
- Clustering Coefficient
- Assortativity
- Centralities (Degree, Betweenness, Eigenvector)
- Avg Path Length (if connected)

### 🔍 Filtering
- Filter nodes by centrality measures (with min/max range)

### 🧩 Community Detection
- Louvain Method
- Girvan-Newman Algorithm

### 📈 Evaluation & Comparison
- Modularity, Conductance, NMI
- Louvain vs. Girvan-Newman comparison

### 🔗 Link Analysis
- PageRank
- Betweenness Centrality

---

## 🖼️ Demo


- ![Image](https://github.com/user-attachments/assets/dd5f6f02-eca1-48bb-ad5f-54ca7a9a0af8)
- ![Image](https://github.com/user-attachments/assets/0f9f5d67-ab2b-46a7-b161-b41fd002356b)
- ![Image](https://github.com/user-attachments/assets/faa3eb1d-a9c8-401d-847c-7380d800a18d)
- ![Image](https://github.com/user-attachments/assets/4725d9e5-2d48-4d5b-8368-c033bab38ff6)


---

## 🚀 How to Run

```bash
pip install networkx pyvis python-louvain scikit-learn
python main.py

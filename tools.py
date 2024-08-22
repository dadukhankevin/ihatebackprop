import networkx as nx
import matplotlib.pyplot as plt


def visualize(network):
    G = nx.DiGraph()

    # Add nodes
    for i in range(network.length):
        if i < network.input_size:
            node_type = 'input'
        elif i >= network.length - network.output_size:
            node_type = 'output'
        else:
            node_type = 'hidden'
        G.add_node(i, type=node_type)

    # Add edges
    for i, connections in enumerate(network.connections):
        for j in connections:
            G.add_edge(i, i + j)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use a hierarchical layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Adjust positions to create layers
    layer_spacing = 1
    for node, (x, y) in pos.items():
        if G.nodes[node]['type'] == 'input':
            pos[node] = (x, 2 * layer_spacing)
        elif G.nodes[node]['type'] == 'output':
            pos[node] = (x, 0)
        else:
            pos[node] = (x, y + layer_spacing)

    # Create a color gradient for hidden nodes
    n_hidden = network.length - network.input_size - network.output_size
    hidden_cmap = plt.get_cmap('viridis')
    hidden_colors = [hidden_cmap(i / n_hidden) for i in range(n_hidden)]

    # Draw nodes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if G.nodes[n]['type'] == 'input':
            node_colors.append('black')
            node_sizes.append(700)
        elif G.nodes[n]['type'] == 'output':
            node_colors.append('orange')
            node_sizes.append(700)
        else:
            node_colors.append(hidden_colors[n - network.input_size])
            node_sizes.append(500)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Draw edges with varying transparency based on source node
    edge_colors = []
    for u, v in G.edges():
        if G.nodes[u]['type'] == 'input':
            edge_colors.append('red')
        elif G.nodes[u]['type'] == 'output':
            edge_colors.append('green')
        else:
            edge_colors.append(hidden_colors[u - network.input_size])

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True,
                           arrowsize=20, alpha=0.3, width=1.5, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add a title and remove axis
    ax.set_title("EnigmaNetwork Visualization", fontsize=20, fontweight='bold')
    ax.axis('off')

    # Add a colorbar for hidden nodes
    sm = plt.cm.ScalarMappable(cmap=hidden_cmap, norm=plt.Normalize(vmin=0, vmax=n_hidden))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Hidden Node Depth', orientation='horizontal',
                        fraction=0.05, pad=0.1, aspect=30)
    cbar.set_label('Hidden Node Depth', fontsize=12, fontweight='bold')

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_connection_density(network):
    densities = []
    for i in range(network.length):
        possible_connections = network.length - i - 1
        actual_connections = len(network.connections[i])
        density = actual_connections / possible_connections if possible_connections > 0 else 0
        densities.append(density)

    plt.figure(figsize=(10, 6))
    plt.plot(range(network.length), densities, marker='o')
    plt.axvline(x=network.input_size - 1, color='g', linestyle='--', label='Input Nodes')
    plt.axvline(x=network.length - network.output_size, color='r', linestyle='--', label='Output Nodes')
    plt.xlabel('Node Index')
    plt.ylabel('Connection Density')
    plt.title('Connection Density Across Network')
    plt.legend()
    plt.grid(True)
    plt.show()
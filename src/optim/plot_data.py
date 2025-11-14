import matplotlib.pyplot as plt
import networkx as nx
import wandb

def plot_markov_chain(true_transitions: dict, name:str, plot_path: str, enable_wandb: None, wandb_path: str) -> None:
    G = nx.MultiDiGraph()
    
    for key, value in true_transitions.items():
        source, _, target = key.partition(" -> ")
        G.add_edge(source, target, label=f"{source}->{target}:{value:.2f}")
    
    pos = nx.circular_layout(G)

    # Create a figure for the plot
    plt.figure(figsize=(10, 12))

    # Draw nodes manually using matplotlib
    for node, (x, y) in pos.items():
        plt.scatter(x, y, s=700, c='skyblue', zorder=2)
        plt.text(x, y, s=node, horizontalalignment='center', verticalalignment='center', fontsize=6, zorder=3)
        

    # Manually draw curved edges with matplotlib arcs and annotate user-specified labels
    for (u, v, key, data) in G.edges(data=True, keys=True):
        print(u, v, key, data)
        # Compute the positions of the nodes
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Define the arc curvature (increase key scale to differentiate more)
        rad = 0.3 * (key + 0.4)  # Increase curvature to avoid overlap

        # Draw the curved arrow with a visible head
        plt.annotate("",
                    xy=(x2, y2), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=2",
                                    color="black",
                                    lw=1.5,
                                    shrinkA=5, shrinkB=5,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=1)

        # Compute the midpoint for labeling the edge
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        label_offset_x = rad * 1 * (y2 - y1)  # Offset perpendicular to edge direction
        label_offset_y = rad * 1 * (x1 - x2)
        xm += label_offset_x  # Adjusting x offset based on curvature
        ym += label_offset_y  # Adjusting y offset based on curvature

        # Annotate the edge with the custom label provided by the user
        plt.text(xm, ym, f"{data['label']}", fontsize=12, color='red', zorder=2)

    # Set title and remove axes
    plt.title("Markov Chain", fontsize=15)
    plt.axis('off')
    
    plt.savefig(f"{plot_path}/{name}.png")
    plt.close()
    
    if enable_wandb:
        wandb.log({f"{wandb_path}/{name}": wandb.Image(f"{plot_path}/{name}.png")})
import matplotlib.pyplot as plt
import numpy as np

def visualize_membership_maps(U, original_shape, algorithm_name="Fuzzy C-Means", cmap='gray'):
    """
    Visualizza le probabilità di appartenenza per ogni cluster separatamente.
    
    Args:
        U: Matrice (N_pixel, N_cluster) con valori tra 0 e 1.
        original_shape: Tupla (H, W) per ricostruire l'immagine.
        algorithm_name: Stringa per il titolo del grafico.
    """
    n_clusters = U.shape[1]
    H, W = original_shape
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    
    if n_clusters == 1: axes = [axes]
    
    print(f"--- Visualizzazione Membership Maps: {algorithm_name} ---")
    
    images = []
    for j in range(n_clusters):
        membership_map = U[:, j].reshape(H, W).squeeze() 
        
        ax = axes[j]
        im = ax.imshow(membership_map, cmap=cmap, vmin=0, vmax=1)
        
        ax.set_title(f"Cluster {j+1}\n(Probabilità)", fontsize=12)
        ax.axis('off')
        images.append(im)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(images[0], cax=cbar_ax)
    
    plt.suptitle(f"Mappe di Appartenenza - {algorithm_name}", fontsize=16)
    plt.subplots_adjust(right=0.9) 
    plt.show()
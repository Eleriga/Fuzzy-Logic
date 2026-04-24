import numpy as np
from sklearn.metrics import silhouette_score

def xie_beni_index(X, U, centers, m=2.0):
    """
    Calcola l'indice di Xie-Beni (XB).
    XB = (Numeratore: Compattezza) / (Denominatore: Separazione)
    
    Minore è il valore, migliore è il clustering.
    """
    # X: (N_samples, N_features)
    # U: (N_samples, N_clusters)
    # centers: (N_clusters, N_features)
    
    X = np.asarray(X)
    U = np.asarray(U)
    centers = np.asarray(centers)
    N = X.shape[0]

    # 1. Calcolo della distanza quadrata di ogni punto da ogni centroide
    # Usiamo un trucco di broadcasting per evitare loop lenti
    # (N, 1, Features) - (1, C, Features) -> (N, C, Features)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) # (N, C)

    # 2. Numeratore: Somma pesata delle distanze (Compattezza)
    # Pesa le distanze con l'appartenenza u_ij^m
    compactness = np.sum((U ** m) * dist_sq)

    # 3. Denominatore: Distanza minima tra i centroidi (Separazione)
    # Calcoliamo tutte le distanze tra centroidi e prendiamo la minima (non zero)
    center_diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    center_dist_sq = np.sum(center_diff**2, axis=2)
    
    # Sostituiamo gli zeri sulla diagonale con infinito per trovare il minimo reale
    np.fill_diagonal(center_dist_sq, np.inf)
    min_separation = np.min(center_dist_sq)

    # Formula finale XB
    xb = compactness / (N * min_separation)
    return xb

def calculate_silhouette(X, labels, sample_size=5000):
    """
    Calcola il Silhouette Score.
    Poiché calcolarlo su tutti i pixel è lentissimo (O(N^2)),
    facciamo un campionamento random.
    
    Valori vicini a 1: Ottimo
    Valori vicini a -1: Pessimo
    """
    if X.shape[0] > sample_size:
        # Prende indici a caso
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
        
    score = silhouette_score(X_sample, labels_sample)
    return score
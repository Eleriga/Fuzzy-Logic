# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="8AhoEGpj2emd"
# # Color quantization

# %% [markdown] id="D1SlGCfJ2gy-"
# Importing libraries

# %% id="JHBx7gkN3lUD" executionInfo={"status": "ok", "timestamp": 1768578628732, "user_tz": -60, "elapsed": 8503, "user": {"displayName": "Eleonora Rigamonti", "userId": "18015720881706151954"}} outputId="ba36e718-5729-42d2-aa14-bc75fbdcfdea" colab={"base_uri": "https://localhost:8080/"}
# !pip install fuzzy-c-means

# %% id="V51ouGQ92bIT"
import numpy as np
from PIL import Image
from fcmeans import FCM
import urllib.request

# %% [markdown] id="k_i9UVjy2odb"
# Getting and rescaling the image

# %% id="AIC2OQBu2qOL" executionInfo={"status": "ok", "timestamp": 1768578629999, "user_tz": -60, "elapsed": 1211, "user": {"displayName": "Eleonora Rigamonti", "userId": "18015720881706151954"}} outputId="612fa7c7-1cd9-4341-d232-574ceb812770" colab={"base_uri": "https://localhost:8080/", "height": 459}
urllib.request.urlretrieve('https://genovese.di.unimi.it/mri_1.jpg', 'image.jpg')
image = Image.open('image.jpg')           # read image (oarsman at https://omadson.github.io/photos/)
N, M = image.size                         # get the number of columns (N) and rows (M)
image                                     # show resized image

# %% [markdown] id="kXEgq-VX2uTv"
# Transforming image into a data set

# %% id="MK5eiGqN2u_r"
X = (
    np.asarray(image)                              # convert a PIL image to np array
    .reshape((N*M, 1))                             # reshape the image to convert each pixel to an instance of a data set
)

# %% [markdown] id="w2WrG0n92xi8"
# Creating and fitting the model

# %% id="5NQFwdtc2yxA"
fcm = FCM(n_clusters=4, m=2)                      # create a FCM instance with 10 clusters
fcm.fit(X)                                         # fit the model

# %% [markdown] id="1j4Be5g020OD"
# Pixel quantization

# %% id="OE_FfzVN20m4"
U = fcm.u                     # (n_pixel, n_clusters)
centers = fcm.centers        # (n_clusters, 1)

# Media pesata fuzzy
transformed_X_fuzzy = U @ centers    # prodotto matriciale
transformed_X_fuzzy.shape = (N*M, 1)


# %% [markdown] id="-46pVDtU23V3"
# Converting and saving image

# %% id="GX8JsdQj231d"
# quantized_array = (
#     transformed_X
#     .astype('uint8')                               # convert data points into 8-bit unsigned integers
#     .reshape((M, N))                            # reshape image
# )

# quantized_image = Image.fromarray(np.asarray(quantized_array))   # convert array into a PIL image object
# quantized_image.save('image_result.bmp') # save image
quantized_array_fuzzy = (
    transformed_X_fuzzy
    .astype(np.uint8)
    .reshape((M, N))
)

quantized_image_fuzzy = Image.fromarray(quantized_array_fuzzy)


# %% id="WZfu0e3jmcM_"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="RX6zpG8b279T"
# Final result

# %% id="31TNCFLD27cf" executionInfo={"status": "ok", "timestamp": 1768578709337, "user_tz": -60, "elapsed": 287, "user": {"displayName": "Eleonora Rigamonti", "userId": "18015720881706151954"}} outputId="f7bc525c-2419-4a43-8dc2-9ff0e561c294" colab={"base_uri": "https://localhost:8080/", "height": 459}
side_by_side = Image.fromarray(
    np.hstack([
        np.array(image),
        np.array(quantized_image_fuzzy)
    ])
)
side_by_side

# %% id="c_ApwbFnW01f" executionInfo={"status": "ok", "timestamp": 1768578719092, "user_tz": -60, "elapsed": 1741, "user": {"displayName": "Eleonora Rigamonti", "userId": "18015720881706151954"}} outputId="b2c0f3ba-e72a-4eca-a8a5-beba1dd56025" colab={"base_uri": "https://localhost:8080/", "height": 444}
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------
# Load image (grayscale)
# -------------------------
# image = Image.open("image.bmp").convert("L")   # assicurati sia grayscale
# M, N = image.size[1], image.size[0]            # altezza, larghezza

# -------------------------
# Image → dataset
# -------------------------
X = np.asarray(image).reshape((M * N, 1))

# -------------------------
# Hard K-Means
# -------------------------
kmeans = KMeans(
    n_clusters=4,
    random_state=0,
    n_init=10
)

kmeans.fit(X)

labels = kmeans.labels_                 # cluster assegnato a ogni pixel
centers = kmeans.cluster_centers_       # centroidi

# -------------------------
# Quantizzazione
# -------------------------
transformed_X = centers[labels]

quantized_array = (
    transformed_X
    .astype(np.uint8)
    .reshape((M, N))
)

quantized_image2 = Image.fromarray(quantized_array)
quantized_image2.save("image_kmeans.bmp")

# -------------------------
# Visualizzazione
# -------------------------
side_by_side = np.hstack([
    np.array(image),
    np.array(quantized_image2)
])

plt.figure(figsize=(10, 5))
plt.imshow(side_by_side, cmap="gray")
plt.title("Originale (sinistra)  |  K-Means (destra)")
plt.axis("off")
plt.show()


# %% id="L8oE10AIXqeW" executionInfo={"status": "ok", "timestamp": 1768578831835, "user_tz": -60, "elapsed": 624, "user": {"displayName": "Eleonora Rigamonti", "userId": "18015720881706151954"}} outputId="5bbb5774-e678-4f2a-ea41-d2c9a3a4bdfe" colab={"base_uri": "https://localhost:8080/", "height": 317}
side = np.hstack([
    np.array(quantized_image_fuzzy),
    np.array(image),
    np.array(quantized_image2)
])

plt.figure(figsize=(10, 5))
plt.imshow(side, cmap="gray")
plt.title("FCM (sinistra)  |  originale  |  K-Means (destra)")
plt.axis("off")
plt.show()

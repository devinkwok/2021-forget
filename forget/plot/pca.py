import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_metrics(
    plotter,
    metrics_examples_array,
    feature_names=None,
    transpose=False,
    n_components_to_plot=10,
):
    # normalize
    array = metrics_examples_array
    array = array - np.min(array, axis=1).reshape(-1, 1)
    max = np.max(array, axis=1).reshape(-1, 1)
    EPSILON = 1e-9  # to avoid dividing by 0
    array = array / np.maximum(max, EPSILON)
    print("Applying PCA")
    print(
        array.shape,
        np.min(array, axis=0),
        np.max(array, axis=0),
        np.mean(array, axis=0),
    )
    if transpose:
        array = array.T
    if feature_names is None:
        feature_names = [str(i) for i in np.arange(array.shape[1])]
    assert len(array.shape) == 2 and array.shape[1] == len(feature_names), array.shape
    pca = PCA()
    transformed = pca.fit_transform(array)
    # bar plot all explained variances
    print(pca.explained_variance_ratio_)
    data = {i: v for i, v in enumerate(pca.explained_variance_ratio_)}
    plotter.plot_bar(f"pca-{feature_names[0]}-etc_{transpose}_expvar", data)
    # bar plot component weights
    feature_order = np.argsort(pca.components_[0])
    print(pca.components_)
    # components are ordered from smallest to largest
    for i in range(min(n_components_to_plot, pca.n_components_)):
        data = {
            feature_names[j]: pca.components_[-1 * (1 + i)][j] for j in feature_order
        }
        plotter.plot_bar(f"pca-{feature_names[0]}-etc_{transpose}_comp{i}", data)
    # plot first 2 components as scatter plot
    # TODO
    # plotter.plt_scatter(transformed[:, :2])

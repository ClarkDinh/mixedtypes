
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans

try:
    from lib.general_lib import *
    from lib.plot import scatter_plot
except Exception as e:
    from general_lib import *
    # from plot import scatter_plot


def Iso_map(X, params, annot_lbl=None):

    print ("Iso_map is running..")

    
    model = Isomap(n_neighbors=params["n_neighbors"])
    #test = PCA(n_components=self.n_components)
    print ("here", params["n_neighbors"])
    x_after_isomap = model.fit_transform(X=X)

    group_index = KMeans(n_clusters=params["n_cluster"]).fit_predict(x_after_isomap)


    if params["visualize"]:
    	scatter_plot(x=x_after_isomap[:, 0], y=x_after_isomap[:, 1], save_file=params["out_dir"], 
    		x_label='Dimension 1', y_label='Dimension 2', title='Isomap and KMeans clustering',
    		annot_lbl=annot_lbl, lbl=None,
    		mode='scatter', sigma=None, 
			interpolate=False, color='blue',
			ax=None, linestyle='-.', marker='o')

    return group_index, x_after_isomap


def Iso_map_with_lbl(X, save_at, is_plot, annot_lbl, colors='blue'):

    model = Isomap(n_neighbors=30)
    x_transform = model.fit_transform(X=X)
    print (colors)

    if is_plot:
        scatter_plot(x=x_transform[:, 0], y=x_transform[:, 1], save_file=save_at, 
            x_label='Dimension 1', y_label='Dimension 2', title='Isomap with group lablel',
            annot_lbl=annot_lbl, lbl=None,
            mode='scatter', sigma=None, 
            interpolate=False, color=colors,
            ax=None, linestyle='-.', marker='o')
        print ("Save at:", save_at)
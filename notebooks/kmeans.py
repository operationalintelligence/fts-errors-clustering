
def kmeans_preproc(dataset, tks_vec):
    """Take input dataset with Word2Vec representation in tks_vec column and properly format to feed into pyspark.ml.KMeans."""
    from pyspark.ml.feature import VectorAssembler
    vec_assembler = VectorAssembler(inputCols = [tks_vec], outputCol='features')
    dataset = vec_assembler.transform(dataset)
    return(dataset)
    

def train_kmeans(dataset, k, ft_col='features', distance="cosine", initSteps=10,
                 tol=0.0001, maxIter=30, save_path=None, mode="new"):
    """Train K-Means model.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
    k (int): number of clusters
    ft_col (string): name of the features column
    distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
    tol (int): tolerance for kmeans algorithm convergence
    maxIter (int): maximum number of iterations for the kmeans algorithm
    save_path (string): where to save trained kmeans model
    mode ("new" or "overwrite"): whether to save new file or overwrite pre-existing one.

    Returns:
    model_fit (pyspark.ml.clustering.KMeansModel): trained K-Means model
    """
    from pyspark.ml.clustering import KMeans
    
    model = KMeans(featuresCol=ft_col,k=k, initMode='k-means||',
                   initSteps=initSteps, tol=tol, maxIter=maxIter, distanceMeasure=distance)
    model_fit = model.fit(dataset)
    
    if save_path:
        if mode=="overwrite":
            model_fit.write().overwrite().save(save_path)
        else:
            model_fit.save(save_path)

    return(model_fit)

def load_kmeans(save_path):
    """Load K-Means model from save_path."""
    from pyspark.ml.clustering import KMeansModel
    model = KMeansModel.load(save_path)
    return(model)
    
def K_optim(k_list, dataset, tks_vec="message_vector", ft_col="features", distance="cosine", 
            initSteps=10, tol=0.0001, maxIter=30, log_path=None):
    """Train K-Means model for different K values.
    
    -- params:
    k_list (list): grid of K values to try
    dataset (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
    tks_vec (string): name of the word2vec representations column
    ft_col (string): name of the features column
    distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
    initStep (int): number of different random intializations for the kmeans algorithm
    tol (int): tolerance for kmeans algorithm convergence
    maxIter (int): maximum number of iterations for the kmeans algorithm
    log_path (string): where to save optimization stats. Default None (no saving)
    
    Returns:
    res (dict): dictionary with grid of trained models and evaluation metrics. Keys:{"model", "wsse", "silhouette"}
    """
    import time
    import datetime
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pathlib import Path
    
    dataset = kmeans_preproc(dataset, tks_vec)
    
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator(distanceMeasure="cosine")

    kmeans_models = []
    clustering_model = []
    wsse = []
    silhouette = []

    for i, k in enumerate(k_list):
            
        start_time = time.time()
        start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

        clustering_model.append(train_kmeans(dataset, ft_col=ft_col, k=k, distance="cosine",
                                             initSteps=initSteps, tol=tol, maxIter=maxIter))

        # compute metrics   
        wsse.append(clustering_model[i].summary.trainingCost)
        silhouette.append(evaluator.evaluate(clustering_model[i].summary.predictions))
        
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as log:
                log.write("With K={}\n\n".format(k))
                log.write("Started at: {}\n".format(start_time_string))
                log.write("Within Cluster Sum of Squared Errors = " + str(round(wsse[i],4)))
                log.write("\nSilhouette with cosine distance = " + str(round(silhouette[i],4)))

                log.write("\nTime elapsed: {} minutes and {} seconds.".format(int((time.time() - start_time)/60), 
                                                                          int((time.time() - start_time)%60)))
                log.write('\n--'*30 + "\n\n")
        else:
            print("With K={}\n".format(k))
            print("Started at: {}\n".format(start_time_string))
            print("Within Cluster Sum of Squared Errors = " + str(round(wsse[i],4)))
            print("Silhouette with cosine distance = " + str(round(silhouette[i],4)))

            print("\nTime elapsed: {} minutes and {} seconds.".format(int((time.time() - start_time)/60), 
                                                                      int((time.time() - start_time)%60)))
            print('--'*30)
            
    res = {"model": clustering_model, "wsse": wsse, "silhouette": silhouette}
    return(res)

def plot_metrics(results):
    """Plot the trends of evaluation metrics from the output of K_optim."""
    import numpy as np
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(22,6))
    
    k_list = [mod.summary.k for mod in results["model"]]
    best_K_wsse = np.argmin(results["wsse"])
    best_K_silhouette = np.argmax(results["silhouette"])

    _ = plt.subplot(1,2,1)
    _ = plt.plot(k_list, np.log(results["wsse"]),'-D', markevery=[best_K_wsse], markerfacecolor='red', markersize=12)
    _ = plt.xlabel("K")
    _ = plt.ylabel("log(WSSE)")
    _ = plt.xticks(k_list)
    _ = plt.title("Within Groups Sum of Squares")

    _ = plt.subplot(1,2,2)
    _ = plt.plot(k_list, results["silhouette"],'-D', markevery=[best_K_silhouette], markerfacecolor='red', markersize=12)
    _ = plt.xlabel("K")
    _ = plt.ylabel("ASW")
    _ = plt.xticks(k_list)
    _ = plt.title("Average Silhouette Width")
    _ = plt.show()
    return(None)

def get_k_best(results, metric="silhouette"):
    """Return the best K value according to the specified metric."""
    import numpy as np
    from matplotlib import pyplot as plt

    best_K_wsse = results["model"][np.argmin(results["wsse"])].summary.k
    best_K_silhouette = results["model"][np.argmax(results["silhouette"])].summary.k
    
    if metric=="silhouette":
        return(best_K_silhouette)
    elif metric=="wsse":
        return(best_K_wsse)
    else:
        print("Error: wrong metric parameter. Specify \"silhouette\" or \"wsse\".")
        return(None)
    
def merge_predictions(original, predictions, orig_id="msg_id", pred_id="msg_id",
                      out_col_list=["tokens_cleaned", "abstract_tokens", "features", "prediction"]):
    """Merge custering output with original dataset (from hdfs) 
                    containing additional information (e.g. src/dst sites, timestamp, ...).
    
    -- params:
    original (pyspark.sql.dataframe.DataFrame): data frame with dta from hdfs
    predictions (pyspark.sql.dataframe.DataFrame): data frame with clustering results
    orig_id (string): name of the message id column on hdfs data frame
    perd_id (string): name of the message id column on prediction data frame
    out_col_list (list[string]): list of names of the columns of the prediction data frame to append in the output

    Returns:
    merge_df (pyspark.sql.dataframe.DataFrame): merged data frame
    """
    import numpy as np
    import pyspark.sql.functions as F
    
    # create list of output columns names
    ## remove duplicate names from right table
    out_col_list = [predictions[col_name] for col_name in out_col_list if col_name not in original.columns]
    ## extract columns from original dataframe with proper format
    output_columns = [original[col_names] for col_names in original.columns]
    ## put left and right columns together in the desired order
    output_columns.extend(out_col_list)
    
    # join original data with predicted cluster labels
    merge_df = original.join(predictions, original[orig_id]==predictions[pred_id], 
                             how="outer").select(output_columns).orderBy(F.col(orig_id))
    return(merge_df)


def kmeans_predict(dataset, model, pred_mode="static", new_cluster_thresh=None, update_model_path=None):
    """Predict cluster for new observations.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
    model (pyspark.ml.clustering.KMeansModel): re-trained kmeans model
    pred_mode ("static" or "update"): prediction mode: "static" does not allow for creating new clusters
    distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
    new_cluster_thresh (float): distance threshold: if closest centroid is more distant than new_cluster_thresh 
                                then a new cluster is created for the new observation
    update_model_path (string): where to save update kmeans model
    
    Returns:
    pred (pyspark.sql.dataframe.DataFrame): the input data frame with an extra "prediction" column
    """
    if pred_mode not in ["static", "update"]:
        print("""WARNING: invalid param \"pred_mode\". Specify either \"static\" to train load a pre-trained model 
              or \"update\" to train it online.""")
        return(None)
    if pred_mode=="static":
        pred = model.transform(dataset)
    else:
        # take centroids
        # compute distances of each message from each centroid
        # select closest centroid per each meassage
        # initialize new clusters when closest centroid distance is greater than new_cluster_thresh
        # update centroids and points in each cluster
        # save updated model
        update_model_path = "temp_filename" # temporary to avoid accidental overwriting
        model.write().overwrite().save(update_model_path)
        pred = None
    return(pred)
    
def kmeans_inference(original_data, msg_col, id_col, w2v_model_path, tks_vec, ft_col, kmeans_mode, kmeans_model_path,
                     pred_mode="static", new_cluster_thresh=None, #update_model_path=None,
                     distance="cosine", opt_initSteps=10, opt_tol=0.0001, opt_maxIter=30, #K_optim
                     tr_initSteps=200, tr_tol=0.000001, tr_maxIter=100, #train_kmeans
                    ):
    from language_models import w2v_preproc
    from pyspark.ml.clustering import KMeansModel
    import time
    import datetime
    from pyspark.ml.evaluation import ClusteringEvaluator
    """Perform inference on new error messages (Note: only K-Means can be re-trained/updated).
    
    -- params:
    original_data (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    msg_col (string): name of the error string column
    id_col (string): name of the message id column
    model_path (string): path where to load pre-trained word2vec model
    tks_vec (string): name of the word2vec representations column
    ft_col (string): name of the features column
    kmeans_mode ("load" or "train"): kmeans mode: "load" uses pre-trained model, while "train" performs online training
    kmeans_model_path (string): path to pre-trained model (Specify None for re-training)
    pred_mode ("static" or "update"): prediction mode: "static" does not allow for creating new clusters
    new_cluster_thresh (float): distance threshold: if closest centroid is more distant than new_cluster_thresh 
                                then a new cluster is created for the new observation
    distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
    opt_initStep (int): number of different random intializations for the kmeans algorithm in the optimization phase
    opt_tol (int): tolerance for kmeans algorithm convergence in the optimization phase
    opt_maxIter (int): maximum number of iterations for the kmeans algorithm in the optimization phase
    tr_initStep (int): number of different random intializations for the kmeans algorithm in the training phase
    tr_tol (int): tolerance for kmeans algorithm convergence in the training phase
    tr_maxIter (int): maximum number of iterations for the kmeans algorithm in the training phase
    
    Returns:
    original_data (pyspark.sql.dataframe.DataFrame): the input data frame with an extra "prediction" column
    """
    if kmeans_mode not in ["load", "train"]:
        print("""WARNING: invalid param \"kmeans_mode\". Specify either \"load\" to train load a pre-trained model 
              or \"train\" to train it online.""")
        return(None)
    
    original_data = w2v_preproc(original_data, msg_col, id_col, w2v_model_path)

    if kmeans_mode=="load":
        original_data = kmeans_preproc(original_data, tks_vec)
        kmeans_model = KMeansModel.load(kmeans_model_path)
    else:
        # K_optim()
        # initialize a grid of K (number of clusters) values
        k_list = range(2, 10)

        # train for different Ks
        res = K_optim(k_list, dataset=original_data, tks_vec=tks_vec, ft_col=ft_col,
                      distance=distance, initSteps=opt_initSteps, tol=opt_tol, maxIter=opt_maxIter)
        
        k_sil = get_k_best(res, "silhouette")
#         dagnostic = plot_metrics(res)

        # kmeans_model = train_kmeans()
    
        ## print in a diagnostic file -- TO DO
    #     start_time = time.time()
    #     start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    #     print("Started at: {}\n".format(start_time_string))

        if pred_mode=="update":
            save_mode = "overwrite"
            kmeans_model_path = "temp_ciccio"
#         elif kmeans_mode=="load":
#             kmeans_model_path = None
#             save_mode = "new"
#         else:
#             save_mode = "new"
        else:
            kmeans_model_path = None
            save_mode = "new"

        original_data = kmeans_preproc(original_data, tks_vec)
        kmeans_model = train_kmeans(original_data, ft_col=ft_col, k=k_sil, distance=distance,
                                    initSteps=tr_initSteps, tol=tr_tol, maxIter=tr_maxIter,
                                    save_path=kmeans_model_path, mode=save_mode)

        ## print in a diagnostic file -- TO DO

#         # compute metrics   
#         best_wsse = kmeans_model.summary.trainingCost
#         best_silhouette = evaluator.evaluate(kmeans_model.summary.predictions)

#         print("With K={}".format(k_sil))
#         print("Within Cluster Sum of Squared Errors = " + str(round(best_wsse,4)))
#         print("Silhouette with cosine distance = " + str(round(best_silhouette,4)))

#         print("\nTime elapsed: {} minutes and {} seconds.".format(int((time.time() - start_time)/60), 
#                                                                   int((time.time() - start_time)%60)))
#         print('--'*30)
    original_data = kmeans_predict(original_data, kmeans_model, pred_mode=pred_mode, 
                                  new_cluster_thresh=None, update_model_path=kmeans_model_path)
    
    return(original_data)
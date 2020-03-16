def stats_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
            abs_tks_in="tokens_cleaned", abstract=True):
    """Compute frequencies of unique messages aggregated per cluster.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_in (string): name of the column with tokens to be abstracted if abstract is True
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages
    
    Returns:
    stats_summary (pandas.DataFrame): data frame with:
                                    "n_messages" --> number of messages per cluster
                                    "unique_strings" --> number of unique messages per cluster
                                    "unique_patterns" (if abstract==True) --> number of unique abstract messages per cluster
    """
    import pyspark.sql.functions as F
    
    grouped_stats = dataset.groupBy(clust_col)
    if abstract:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                   F.countDistinct(tks_col).alias("unique_strings"),
                                   F.countDistinct(abs_tks_out).alias("unique_patterns"),
                                   ).orderBy("n_messages", ascending=False)
#                                    ).orderBy("unique_patterns", ascending=False)
    else:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                   F.countDistinct(tks_col).alias("unique_strings"),
                                   ).orderBy("n_messages", ascending=False)
#                                    ).orderBy("unique_strings", ascending=False)
        
    return(stats_summary.toPandas())

def pattern_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_tokens",
                    abs_tks_in="tokens_cleaned", abstract=True, n_mess=3, original=None, n_src=3, n_dst=3, 
                    src_col=None, dst_col=None):
    """Compute top n_mess messages aggregated per cluster.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_in (string): name of the column with tokens to be abstracted if abstract is True
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages
    n_mess (int): number of most frequent patterns to retain
    original (pyspark.sql.dataframe.DataFrame): data frame with hdfs data for enriched summary.
                                                Default None (no additional information is showed)
    n_src (int): number of most frequent source sites to retain  -- Default None (TO DO)
    n_dst (int): number of most frequent destination sites to retain  -- Default None (TO DO)
    src_col (string): name of the source site column in the original data frame  -- Default None (TO DO)
    dst_col (string): name of the destination site column in the original data frame  -- Default None (TO DO)
    
    Returns:
    patterns_summary (pandas.DataFrame): data frame with:
                                    "top_{n_mess}" --> dictionary with top n_mess patterns per cluster 
                                            (Keys are ["msg": contains the pattern, "n": relative frequency in the cluster] 
                                    "top_{n_src}" --> dictionary with top n_src source sites per cluster 
                                            (Keys are ["src": contains the source, "n": relative frequency in the cluster] 
                                    "top_{n_dst}" --> dictionary with top n_mess per cluster 
                                            (Keys are ["dst": contains the destination, "n": relative frequency in the cluster] 
    """
    import pandas as pd
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    
    # extract top N patterns per each cluster
    if abstract:
        msg_col = (abs_tks_out, "unique_patterns")
    else:
        msg_col = (tks_col, "unique_strings")
    
#     if original:
#         src_col = [col_name for col_name in original.columns if "src" in col_name][0]
#         dst_col = [col_name for col_name in original.columns if "dst" in col_name][0]
    
    # groupby cluster_id and patterns and count groups frequencies
    grouped_patterns = dataset.groupBy(clust_col, msg_col[0]).agg(F.count("*").alias(msg_col[1])
                                ).orderBy(clust_col, msg_col[1], ascending=[True, False])
    
    window_pattern = Window.partitionBy(clust_col).orderBy(F.col(msg_col[1]).desc(),
                                                   F.col(msg_col[0]))
    if original:
        grouped_patterns_src = dataset.groupBy(clust_col, src_col).agg(F.count("*").alias("src_sites")
                                ).orderBy(clust_col, "src_sites", ascending=[True, False])
        grouped_patterns_dst = dataset.groupBy(clust_col, dst_col).agg(F.count("*").alias("dst_sites")
                                ).orderBy(clust_col, "dst_sites", ascending=[True, False])
        window_src = Window.partitionBy(clust_col).orderBy(F.col("src_sites").desc(),
                                                       F.col(src_col))
        window_dst = Window.partitionBy(clust_col).orderBy(F.col("dst_sites").desc(),
                                                       F.col(dst_col))
        grouped_patterns = grouped_patterns.select('*', F.rank().over(window_pattern).alias('rank_pattern')
                                                  ).filter(F.col('rank_pattern') <= n_mess)
        grouped_patterns_src = grouped_patterns_src.select('*', F.rank().over(window_src).alias('rank_src')
                                                  ).filter(F.col('rank_src') <= n_src) 
        grouped_patterns_dst = grouped_patterns_dst.select('*', F.rank().over(window_dst).alias('rank_dst')
                                                  ).filter(F.col('rank_dst') <= n_dst)
        
        
#         columns = [F.col(col_name) for col_name in grouped_patterns]
#         grouped_patterns = grouped_patterns.join(grouped_patterns_src,
#                                                  grouped_patterns[clust_col]==grouped_patterns_src[clust_col], 
#                                                  how="outer").select()
#         grouped_patterns = grouped_patterns.join(grouped_patterns_dst,
#                                                  grouped_patterns[clust_col]==grouped_patterns_dst[clust_col], 
#                                                  how="outer").select(grouped_patterns[clust_col])
#         return(grouped_patterns)#, grouped_patterns_src, grouped_patterns_dst)

    else:
        grouped_patterns = grouped_patterns.select('*', F.rank().over(window_pattern).alias('rank_pattern')) \
        .filter(F.col('rank_pattern') <= n_mess) 
    
    # take top n_mess patterns for each cluster/pattern group
    col_out = []
    clust_labels = []
    for clust_id in grouped_patterns.select(clust_col).distinct().collect():
        temp = grouped_patterns.filter(F.col(clust_col)==clust_id[clust_col]).select(
            msg_col[0], msg_col[1]).collect()

        row_out = []
        for row in temp:
            row_out.append({"msg": " ".join(row[msg_col[0]]),#[1:-1].split(",")),
                            "n": row[msg_col[1]]})
        clust_labels.append(clust_id[clust_col])
        col_out.append(row_out)
    
    patterns_summary = pd.DataFrame({"top_{}_msg".format(n_mess): col_out}, index=clust_labels)

    if original:
        # take top n_src source sites for each cluster/src_site group
        col_out = []
        clust_labels = []
        for clust_id in grouped_patterns_src.select(clust_col).distinct().collect():
            temp = grouped_patterns_src.filter(F.col(clust_col)==clust_id[clust_col]).select(
                src_col, "src_sites").collect()

            row_out = []
            for row in temp:
                row_out.append({"src": row[src_col],
                                "n": row["src_sites"]})
            clust_labels.append(clust_id[clust_col])
            col_out.append(row_out)

        src_summary = pd.DataFrame({"top_{}_src".format(n_src): col_out}, index=clust_labels)

        # take top n_dst destination sites for each cluster/dst_site group
        col_out = []
        clust_labels = []
        for clust_id in grouped_patterns_dst.select(clust_col).distinct().collect():
            temp = grouped_patterns_dst.filter(F.col(clust_col)==clust_id[clust_col]).select(
                dst_col, "dst_sites").collect()

            row_out = []
            for row in temp:
                row_out.append({"dst": row[dst_col],
                                "n": row["dst_sites"]})
            clust_labels.append(clust_id[clust_col])
            col_out.append(row_out)

        dst_summary = pd.DataFrame({"top_{}_dst".format(n_dst): col_out}, index=clust_labels)

        # merge summary data frames
        patterns_summary = pd.merge(patterns_summary, src_summary, how='outer',
                    left_index=True, right_index=True)
        patterns_summary = pd.merge(patterns_summary, dst_summary, how='outer',
                    left_index=True, right_index=True)        
    return(patterns_summary)

    
def summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
            abs_tks_in="tokens_cleaned", abstract=True, n_mess=3, wrdcld=False,  #stats_summary
            original=None, n_src=3, n_dst=3, src_col=None, dst_col=None, data_id="msg_id", orig_id="msg_id", #patterns_summary
           ):
    """Return summary statistics aggregated per cluster.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_in (string): name of the column with tokens to be abstracted if abstract is True
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages
    n_mess (int): number of most frequent patterns to retain
    wrdcld (bool): whether to produce word cloud fr visualization of clusters content    
    original (pyspark.sql.dataframe.DataFrame): data frame with hdfs data for enriched summary.
                                                Default None (no additional information is showed)
    n_src (int): number of most frequent source sites to retain  -- Default None (TO DO)
    n_src (int): number of most frequent destination sites to retain  -- Default None (TO DO)
    src_col (string): name of the source site column in the original data frame  -- Default None (TO DO)
    dst_col (string): name of the destination site column in the original data frame  -- Default None (TO DO)
    data_id (string): name of the message id column in the dataset data frame
    orig_id (string): name of the message id column in the original data frame

    Returns:
    summary_df (pandas.DataFrame): merged data frame with stats_summary and patterns_summary
    """
    import pandas as pd
    from abstraction_utils import abstract_params
    
        # compute quantitative stats of the clusters
    if abstract:
        dataset = abstract_params(dataset, tks_col=abs_tks_in, out_col=abs_tks_out)
        
    if original:
        dataset = original.join(dataset, original[orig_id]==dataset[data_id], 
                             how="outer")#.select(output_columns)
    # first compute quantitative stats of the clusters
    stats = stats_summary(dataset, clust_col=clust_col, tks_col=tks_col, abs_tks_out=abs_tks_out,
            abs_tks_in=abs_tks_in, abstract=abstract)
        
    # second extract top N most frequent patterns
    patterns = pattern_summary(dataset, clust_col=clust_col, tks_col=tks_col, abs_tks_out=abs_tks_out,
                               abs_tks_in=abs_tks_in, abstract=abstract, n_mess=n_mess, original=original, 
                               n_src=n_src, n_dst=n_dst, src_col=src_col, dst_col=dst_col)
        
    # finally combine the stats and patterns summary
    summary_df = pd.merge(stats, patterns, how='outer',
                    left_on=clust_col, right_on=patterns.index).set_index(clust_col)
    
    # Add percentage stat in top patterns/src/dst columns
    for idx in summary_df.index:
        tot_clust = summary_df.loc[idx].n_messages
        cols = [col for col in summary_df.columns if "top" in col]
        for col in  cols:
            output = []
            for top_enties in summary_df[col].loc[idx]:
                top_enties["n_perc"] = round(top_enties["n"]/tot_clust, 4)
                output.append(top_enties)
            summary_df[col].loc[idx] = output
    
    # tokens cloud
    if wrdcld:
        tokens_cloud(dataset, msg_col=abs_tks_out, clust_col=clust_col)
    return(dataset, summary_df)

def tokens_cloud(dataset, msg_col, clust_col="prediction", save_path=None,
                figsize=(8,4), width=800, height=400, bkg_col="white", min_font_size=11):
    """Return summary statistics aggregated per cluster.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    msg_col (string): name of the tokens lists column
    clust_col (string): name of the cluster prediction column
    save_path (string): where to save output figures. Defaule None (no saving)
    figsize (tuple(int, int)): figure size
    width (int): width of word clouds
    height (int): height of word clouds
    bkg_col (string): background color of word clouds
    min_font_size (int): fontsize for the least commond tokens

    Returns: None
    """
    import wordcloud as wrdcld
    import matplotlib
    import pyspark.sql.functions as F
    from matplotlib import pyplot as plt
    from abstraction_utils import abstract_params

    for clust_id in dataset.select(clust_col).distinct().collect():
        cluster_messages = dataset.filter(F.col(clust_col)==clust_id[clust_col]).select(msg_col).collect()
        if type(cluster_messages[0][msg_col]) == type([]):
            cluster_messages = [tkn.strip() for tks_msg in cluster_messages for tkn in tks_msg[msg_col]]
        else:
            cluster_messages = [msg[msg_col].strip() for msg in cluster_messages]
        
        # Create and generate a word cloud image:
        wordcloud = wrdcld.WordCloud(width=width, height=height, background_color=bkg_col, 
                                    min_font_size=min_font_size, 
                                    colormap=matplotlib.cm.inferno).generate(" ".join(cluster_messages))

        # Display the generated image:
        fig = plt.figure(figsize=figsize)
        plt.title("CLUSTER {}".format(clust_id[clust_col]))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        if save_path:
            fig.savefig("{}/Cluster{}.png".format(save_path, clust_id[clust_col]), format='png', bbox_inches='tight')

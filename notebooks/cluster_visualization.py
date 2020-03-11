def stats_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
            abs_tks_in="tokens_cleaned", abstract=True):
    import pyspark.sql.functions as F
    
    grouped_stats = dataset.groupBy(clust_col)
    if abstract:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                   F.countDistinct(tks_col).alias("unique_strings"),
                                   F.countDistinct(abs_tks_out).alias("unique_patterns"),
                                   ).orderBy("unique_patterns", ascending=False)
    else:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                   F.countDistinct(tks_col).alias("unique_strings"),
                                   ).orderBy("unique_strings", ascending=False)
        
    return(stats_summary.toPandas())

def pattern_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_tokens",
                    abs_tks_in="tokens_cleaned", abstract=True, n_mess=3, original=None, n_src=3, n_dst=3, 
                    src_col=None, dst_col=None):
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
        grouped_patterns_src = dataset.groupBy(clust_col, msg_col[0], src_col).agg(F.count("*").alias("src_sites")
                                ).orderBy(clust_col, "src_sites", ascending=[True, False])
        grouped_patterns_dst = dataset.groupBy(clust_col, msg_col[0], dst_col).agg(F.count("*").alias("dst_sites")
                                ).orderBy(clust_col, "dst_sites", ascending=[True, False])
        window_src = Window.partitionBy(clust_col).orderBy(F.col("src_sites").desc(),
                                                       F.col(msg_col[0]), F.col(src_col))
        window_dst = Window.partitionBy(clust_col).orderBy(F.col("dst_sites").desc(),
                                                       F.col(msg_col[0]), F.col(dst_col))
        grouped_patterns = grouped_patterns.select('*', F.rank().over(window_pattern).alias('rank_pattern')
                                                  ).filter(F.col('rank_pattern') <= n_mess)
        grouped_patterns_src = grouped_patterns_src.select('*', F.rank().over(window_src).alias('rank_src')
                                                  ).filter(F.col('rank_src') <= n_src) 
        grouped_patterns_dst = grouped_patterns_dst.select('*', F.rank().over(window_dst).alias('rank_dst')
                                                  ).filter(F.col('rank_dst') <= n_dst)
        
        return(grouped_patterns, grouped_patterns_src, grouped_patterns_dst)

    else:
        grouped_patterns = grouped_patterns.select('*', F.rank().over(window_pattern).alias('rank_pattern')) \
        .filter(F.col('rank_pattern') <= n_mess) 
    
    # take top N patterns for each cluster/pattern group
    col_out = []
    clust_labels = []
    for clust_id in grouped_patterns.select(clust_col).distinct().collect():
        if original:
            temp = grouped_patterns.filter(F.col(clust_col)==clust_id[clust_col]).select(
                msg_col[0], msg_col[1], src_col, dst_col).collect()
        else:
            temp = grouped_patterns.filter(F.col(clust_col)==clust_id[clust_col]).select(
                msg_col[0], msg_col[1]).collect()

        row_out = []
        for row in temp:
            row_out.append({"msg": " ".join(row[msg_col[0]]),#[1:-1].split(",")),
                            "n": row[msg_col[1]]})
        clust_labels.append(clust_id[clust_col])
        col_out.append(row_out)
    
    patterns_summary = pd.DataFrame({"top_3": col_out}, index=clust_labels)
    return(patterns_summary)

    
def summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
            abs_tks_in="tokens_cleaned", abstract=True, n_mess=3, wrdcld=False, original=None, n_src=3, n_dst=3,
            data_id="msg_id", orig_id="msg_id", src_col=None, dst_col=None):
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
                               n_src=3, n_dst=3, src_col=src_col, dst_col=dst_col)
        
    # finally combine the stats and patterns summary
    summary_df = pd.merge(stats, patterns, how='outer',
                    left_on=clust_col, right_on=patterns.index).set_index(clust_col)
    
    if wrdcld:
        tokens_cloud(dataset, msg_col=abs_tks_out, clust_col=clust_col)
    return(dataset, summary_df)
#     return(patterns[0],patterns[1], patterns[2])

def tokens_cloud(dataset, msg_col, clust_col="prediction", save_path=None,
                figsize=(8,4), width=800, height=400, bkg_col="white", min_font_size=11):
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

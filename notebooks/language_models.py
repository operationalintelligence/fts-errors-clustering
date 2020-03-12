def tokenizer(dataset, err_col, id_col="msg_id"):
    """Take input message and split it into tokens.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    err_col (string): name of the error string column
    id_col (string): name of the message id column
    
    Returns:
    vector_data (pyspark.sql.dataframe.DataFrame): data frame with id_col, err_col and additional tokenization steps:
                                     corrected_message --> string with corrected urls 
                                     tokens --> list of tokens taken from corrected_message
                                     tokens_cleaned --> list of tokens cleaned from punctuation and empty entries
                                     stop_token --> list of tokens after removing common english stopwords
                                     stop_token_1 --> list of tokens after removing custom stopwords, i.e. ["", ":", "-", "+"]
    """
    import pyspark.sql.functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType, ArrayType
    from pyspark.ml.feature import Tokenizer, StopWordsRemover
    from pyspark.ml import Pipeline
    from text_parsing_utils import split_urls, clean_tokens
    
    # transform in user defined function
    split_urls_udf = udf(split_urls, StringType())

    # split urls appropriately
    test_data = dataset.select(id_col, err_col).withColumn("corrected_message", split_urls_udf(err_col))
    
    # split text into tokens
    tokenizer = Tokenizer(inputCol="corrected_message", outputCol="tokens")
    vector_data = tokenizer.transform(test_data)

    # transform in user defined function
    clean_tokens_udf = udf(lambda entry: clean_tokens(entry, custom_split=True), ArrayType(StringType()))

    # clean tokens
    vector_data = vector_data.withColumn("tokens_cleaned", clean_tokens_udf("tokens"))

    # remove stop (common, non-relevant) words
    stop_remove = StopWordsRemover(inputCol="tokens_cleaned", outputCol="stop_token")
    stop_remove1 = StopWordsRemover(inputCol="stop_token", outputCol="stop_token_1", stopWords=["", ":", "-", "+"])

    data_prep_pipeline = Pipeline(stages = [stop_remove, stop_remove1])

    pipeline_executor = data_prep_pipeline.fit(vector_data)
    vector_data = pipeline_executor.transform(vector_data)

    return(vector_data)

def train_w2v(dataset, tks_col="stop_token_1", id_col="msg_id", out_col='message_vector', 
              vec_size=3, min_count=1, save_path=None, mode="new"):
    """Train Word2Vec model on the input tokens column.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    tks_col (string): name of the column containing the lists of tokens to feed into the word2vec model
    id_col (string): name of the message id column
    out_col (string): name of the output column for the word2vec vector representation of the messages
    vec_size (int): dimension of the word2vec embedded space
    min_count (int): minimum frequency for tokens to be considered in the training
    save_path (string): path where to save the trained model. Default is None (no saving)
    mode ("new" or "overwrite"): whether to save new file or overwrite pre-existing one.
    
    Returns:
    model (pyspark.ml.feature.Word2VecModel): trained Word2vec model
    """
    from pyspark.ml.feature import Word2Vec

    # intialise word2vec
    word2vec = Word2Vec(vectorSize = vec_size, minCount = min_count, inputCol = tks_col, outputCol = out_col)

    train_data = dataset.select(id_col, tks_col)
    model = word2vec.fit(train_data)
    
    if save_path:
        if mode=="overwrite":
            model.write().overwrite().save(save_path)
        else:
            model.save(save_path)
    return(model)

def load_w2v(model_path):
    """Load Word2Vec model from model_path."""
    
    from pyspark.ml.feature import Word2VecModel
    w2vec_model = Word2VecModel.load(model_path)
    return(w2vec_model)

def w2v_preproc(original_data, msg_col, id_col, model_path):
    """Take input dataset as extracted from hdfs and compute Word2Vec representation.
    
    -- params:
    original_data (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    msg_col (string): name of the error string column
    id_col (string): name of the message id column
    model_path (string): path where to load pre-trained word2vec model
    
    Returns:
    original_data (pyspark.ml.feature.Word2VecModel): the original data with an extra 
                    "message_vector" column with word2vec embedding
    """
    original_data = tokenizer(original_data, err_col=msg_col, id_col=id_col)
    w2v_model = load_w2v(model_path)
    original_data = w2v_model.transform(original_data)
    return(original_data)
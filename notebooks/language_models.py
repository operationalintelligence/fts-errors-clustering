def tokenizer(dataset, err_col, id_col="msg_id"):
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
    from pyspark.ml.feature import Word2VecModel
    w2vec_model = Word2VecModel.load(model_path)
    return(w2vec_model)

def w2v_preproc(original_data, msg_col, id_col, model_path):
    original_data = tokenizer(original_data, err_col=msg_col, id_col=id_col)
    w2v_model = load_w2v(model_path)
    original_data = w2v_model.transform(original_data)
    return(original_data)
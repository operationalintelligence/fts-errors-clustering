def deal_with_urls(url):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme:
        if parsed.path and parsed.netloc:
            res = ' '.join([parsed.scheme + "://" + parsed.netloc, parsed.path])
        elif parsed.path:
            res = parsed.scheme + ":" + parsed.path
        else:
            res = parsed.scheme + ":"
    else:
        res = parsed.path
    return(res)

def split_urls(string):
    '''Take a string and split url into netloc + path'''
    from urllib.parse import urlparse
    tks = [deal_with_urls(x) for x in string.split()]
    return(' '.join(tks))


def split_concatenation_errors(string, split_char="-"):
    """"""   
    if split_char in string:
        try:
            int(string.split(split_char)[0])
            number = string.split(split_char)[0]
            literal = string.replace(number+split_char, "")
            return([number, literal])
        except ValueError:
            return(string)
    else:
        return(string)

def clean_tokens(entry, custom_split=False):
    """Remove punctuation at the end of tokens and discard empty tokens"""
    def flatten(ul):
        """"""
        fl = []
        for i in ul:
            if type(i) is list:
                fl += flatten(i)
            else:
                fl += [i]
        return fl
    
    if custom_split:
        tks_cleaned = [split_concatenation_errors(tks.strip(":;,.- ")) 
                       for tks in entry if tks!=""]
        tks_cleaned = flatten(tks_cleaned)
    else:
        tks_cleaned = [tks.strip(":;,.- ") for tks in entry if tks!=""]
    return(tks_cleaned)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

def is_url(string):
    if string.find("://") != -1:
        return(True)
    return(False)
    

def is_ipv6(string):
    import numpy as np
    is_column = [1 if x==":" else 0 for x in string]
    if np.sum(is_column) >= 3 and string.find("/") == -1:
        return(True)
    return(False)

def is_address(string):
    import numpy as np
    is_point = [1 if x=="." else 0 for x in string]
    if np.sum(is_point) >= 2 and string.find("/") == -1:
        return(True)
    return(False)

def is_path(string):
    if string.startswith("path="):
        return(True)
    return(False)

def is_file_path(string):
    if string.startswith("/") and all(char in string[1:] for char in [".", "/"]):
        return(True)
    return(False)

def is_net_param(string):
    if string.startswith("[net="):
        return(True)
    return(False)

def is_filesize_mismatch(string):
    if string.startswith("(expected="):
        res = "$EXPECTED_SIZE"
    elif string.startswith("actual="):
        res = "$ACTUAL_SIZE"
    else:
        res = False
    return(res)

def is_remote_entity(string):
    if string.startswith("(/cn="):
        res = "$EXPECTED_REMOTE_ENTITY"
    elif string.startswith("(/dc="):
        res = "$ACTUAL_REMOTE_ENTITY"
    else:
        res = False
    return(res)
    
def replace_address(string, replace_string=None):
    if not replace_string:
        if is_url(string):
            string = "$URL"
        elif is_ipv6(string):
            string = "$IPv6"
        elif is_address(string):
            string = "$ADDRESS"
        elif is_path(string):
            string = "$PATH"
        elif is_file_path(string):
            string = "$FILE_PATH"
        elif is_net_param(string):
            string = "$NET_PARAM"
        elif is_filesize_mismatch(string):
            string = is_filesize_mismatch(string)
        elif is_remote_entity(string):
            string = is_remote_entity(string)
    else:
        if is_url(string) or is_ipv6(string) or is_address(string) or is_path(string):
            string = replace_string
    return(string)

def replace_IDS(tokens_list, ID_precursors=["transaction", "process", "message", "relation", "database", "tuple"]):
    import re
    msg_pattern = re.compile("<[0-9]+:[0-9]+>")
    precursor_flags = [False]*len(ID_precursors)
    
    res = []
    for tkn in tokens_list:
        
        if True in precursor_flags:
            i = precursor_flags.index(True)
            if ID_precursors[i]=="message" and not msg_pattern.match(tkn):
                res.append(tkn)
                precursor_flags[i]=False
                next
            res.append("${}_ID".format(ID_precursors[i].upper()))
            precursor_flags[i]=False
        else:
            res.append(tkn)
        
        if tkn in ID_precursors:
            precursor_flags[ID_precursors.index(tkn)]=True
    return(res)

def abstract_message(tokens_list):
    '''Take a string and split url into netloc + path'''
    tks = [replace_address(x) for x in tokens_list]
    tks = replace_IDS(tks)
    return(tks)


def post_processing(model, dataset, tks_vec="stop_token_1"):
    import pyspark.sql.functions as F

    # transform in user defined function
    abstract_message_udf = udf(abstract_message, ArrayType(StringType()))
    
    data_subset_no_abs = model.summary.predictions
    data_subset_no_abs = data_subset_no_abs.withColumn("abstract_tokens", abstract_message_udf(tks_vec))

    grouped_pred = data_subset_no_abs.groupBy("prediction")
    grouped_pred_agg = grouped_pred.agg(F.countDistinct("message").alias("n_unique_mess")).orderBy("n_unique_mess", ascending=False)
    grouped_pred_agg.withColumnRenamed("prediction", "cluster_label").show(truncate=False)
    
    return(grouped_pred_agg)
def is_url(string):
    """Check whether input string is an url."""
    if string.find("://") != -1:
        return(True)
    return(False)

def is_ipv6(string):
    """Check whether input string is an ipv6."""
    import numpy as np
    is_column = [1 if x==":" else 0 for x in string]
    if np.sum(is_column) >= 3 and string.find("/") == -1:
        return(True)
    return(False)

def is_address(string):
    """Check whether input string is an address."""
    import numpy as np
    is_point = [1 if x=="." else 0 for x in string]
    if np.sum(is_point) >= 2 and string.find("/") == -1:
        return(True)
    return(False)

def is_path(string):
    """Check whether input string is a path."""
    if string.startswith("path="):
        return(True)
    return(False)

def is_file_path(string):
    """Check whether input string is a file path."""
    if string.startswith("/") and all(char in string[1:] for char in [".", "/"]):
        return(True)
    return(False)

def is_net_param(string):
    """Check whether input string is a network parameter."""
    if string.startswith("[net="):
        return(True)
    return(False)

def is_filesize_mismatch(string):
    """Check whether input string is a mismatch in file size."""
    if string.startswith("(expected="):
        res = "\$EXPECTED_SIZE"
    elif string.startswith("actual="):
        res = "\$ACTUAL_SIZE"
    else:
        res = False
    return(res)

def is_remote_entity(string):
    """Check whether input string is a remote entity."""
    if string.startswith("(/cn="):
        res = "\$EXPECTED_REMOTE_ENTITY"
    elif string.startswith("(/dc="):
        res = "\$ACTUAL_REMOTE_ENTITY"
    else:
        res = False
    return(res)
    
def replace_params(string, replace_string=None):
    """Return inptu string after abstracting url, ipv6, address, path, filepath,
        network parameters, filesize mismatch and remote entity.
    """
    if not replace_string:
        if is_url(string):
            string = "\$URL"
        elif is_ipv6(string):
            string = "\$IPv6"
        elif is_address(string):
            string = "\$ADDRESS"
        elif is_path(string):
            string = "\$PATH"
        elif is_file_path(string):
            string = "\$FILE_PATH"
        elif is_net_param(string):
            string = "\$NET_PARAM"
        elif is_filesize_mismatch(string):
            string = is_filesize_mismatch(string)
        elif is_remote_entity(string):
            string = is_remote_entity(string)
    else:
        if is_url(string) or is_ipv6(string) or is_address(string) or is_path(string):
            string = replace_string
    return(string)

def replace_IDS(tokens_list, ID_precursors=["transaction", "process", "message", "relation", "database", "tuple"]):
    """Replace actual ids for ID_precursors entities with "$ID.""""
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
            res.append("\${}_ID".format(ID_precursors[i].upper()))
            precursor_flags[i]=False
        else:
            res.append(tkn)
        
        if tkn in ID_precursors:
            precursor_flags[ID_precursors.index(tkn)]=True
    return(res)

def abstract_message(tokens_list):
    '''Return input tokens list afterabstracting parameters'''
    tks = [replace_params(x) for x in tokens_list]
    tks = replace_IDS(tks)
    return(tks)

def abstract_params(dataset, tks_col="tokens_cleaned", out_col="abstract_message"):
    """Abstract parameters from a column of tokens lists.
    
    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with at least a column containg lists of tokens
    tks_col (string): name of the tokens lists column
    out_col (string): name of the column where to store abstracted tokens
    
    Returns:
    dataset (pyspark.sql.dataframe.DataFrame): the input dataset with an extra 
                    out_col column with abstracted tokens
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType, ArrayType
    
    # transform in user defined function
    abstract_message_udf = udf(abstract_message, ArrayType(StringType()))
    
    dataset = dataset.withColumn("abstract_tokens", abstract_message_udf("tokens_cleaned"))
    return(dataset)
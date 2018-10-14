from sqlalchemy import create_engine

def sql_str_generator(user, pword, host, dbase):
    """
    Return string use for connection
    """
    return "mysql+pymysql://" + user + ":" + pword + "@" + host + "/" + dbase


connstr = sql_str_generator("root","pass23","localhost","world")
conn = create_engine(connstr).connect()
conn.close()

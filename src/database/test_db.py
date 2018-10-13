import pymysql

def test_schema(cursor):
    query = 'SHOW TABLES'
    cursor.execute(query)

    for table in cursor:
        print(table)

def main():
    #grab credentials
    with open('db_credentials.txt') as file:
        credentials = [x.strip().split(":") for x in file.readlines()]
    for username, password in credentials:
        user = username
        pw = password
    #create connection object
    conn = pymysql.connect(host='ws-db.cxn6r23mlloe.us-east-1.rds.amazonaws.com',user=user, password=pw, db='corpus')
    cursor = conn.cursor()

    test_schema(cursor)
    
main()

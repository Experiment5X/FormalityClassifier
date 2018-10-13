import pymysql

def test_schema(cursor):
    query = 'SHOW TABLES'
    cursor.execute(query)

    for table in cursor:
        print(table)

def test_insert_many(cursor):
    data = [
        ('ThisIsATestWord4', 0.72973),
        ('ThisIsATestWord3', 0.81081)
        ]
    query = "INSERT INTO Word_Formality (word,formality) VALUES (%s, %s)"
    cursor.executemany(query, data)
    cursor.execute("SELECT * from Word_Formality")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    

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
    test_insert_many(cursor)
    conn.commit()
    conn.close()
main()

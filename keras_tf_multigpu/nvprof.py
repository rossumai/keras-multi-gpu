# NVIDIA Profier (nvprof) utilities
#
# - figure total execution time (so that we can sample a tiny bit)
#
# TODO:
# - remove data unnecessary for further analysis (reducing the dataset size)

import sys

import sqlite3

def total_time(conn, table='CUPTI_ACTIVITY_KIND_MEMCPY'):
    c = conn.cursor()
    c.execute('SELECT (1.0 * MAX(end) - 1.0 * MIN(start)) * 1e-9 FROM {}'.format(table))
    total_time = c.fetchone()[0]
    print('total time: %.03f sec' % total_time)

def list_tables(conn):
    c = conn.cursor()
    c.execute('select name from sqlite_master where type="table" and name like "CUPTI%" order by name')
    return [row[0] for row in c.fetchall()]

if __name__ == '__main__':
    db_name = sys.argv[1]
    conn = sqlite3.connect(db_name)
    total_time(conn)
    conn.close()

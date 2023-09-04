import os
import numpy as np
import sqlite3
import pickle
import zlib
from hashlib import md5


class DBSet:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        with self.conn:
            self.conn.execute('CREATE TABLE IF NOT EXISTS objects (hash TEXT PRIMARY KEY, object BLOB)')

    def __len__(self):
        with self.conn:
            return self.conn.execute('SELECT COUNT(*) FROM objects').fetchone()[0]

    def __contains__(self, obj):
        hash_value = md5(pickle.dumps(obj)).hexdigest()
        with self.conn:
            return self.conn.execute('SELECT EXISTS(SELECT 1 FROM objects WHERE hash=?)', (hash_value,)).fetchone()[0]

    def __iter__(self):
        with self.conn:
            for row in self.conn.execute('SELECT object FROM objects'):
                obj = pickle.loads(row[0])
                yield obj

    def add(self, obj):
        byte_obj = pickle.dumps(obj)
        hash_value = md5(pickle.dumps(obj)).hexdigest()
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO objects (hash, object) VALUES (?, ?)', (hash_value, byte_obj))

    def discard(self, obj):
        hash_value = md5(pickle.dumps(obj)).hexdigest()
        with self.conn:
            self.conn.execute('DELETE FROM objects WHERE hash=?', (hash_value,))

    def close(self):
        self.conn.close()

    def __del__(self) -> None:
        self.close()


if __name__ == "__main__":

    dbset = DBSet("hoge.db")
    for i in range(100):
        dbset.add(np.random.randn(100))
        assert len(dbset) == i + 1

    dbset.add(np.zeros(100))
    assert len(dbset) == 101

    dbset.add(np.zeros(100))
    assert len(dbset) == 101

    # print(len(dbset))
    # dbset.add("unko")
    # assert len(dbset) == 1
    # dbset.add("chinko")
    # assert len(dbset) == 2
    # dbset.add("chinko")
    # assert len(dbset) == 2

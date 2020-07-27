import lmdb
from tqdm import tqdm
import os

try:
    from . import treedata_pb2 as pb
except ImportError:
    import treedata_pb2 as pb

def stream_lmdb(root, decode=False):
    env_in = lmdb.open(root,
                          max_readers=1,
                          readonly=True,
                          lock=False,
                          readahead=False,
                          meminit=False,
                          max_dbs=1)

    in_db = env_in.open_db(
        key='cpg-data'.encode('utf-8'), create=False
    )

    with env_in.begin(write=False, db=in_db) as txn1:
        cursor = txn1.cursor()

        for idx, data in tqdm(enumerate(cursor.iternext()), total=txn1.stat()['entries']):

            _, data = data

            if decode:
                tree = pb.AnnotatedTree()
                tree.ParseFromString(data)
                data = tree

            yield idx, data

        cursor.close()


def feed_lmdb(root, stream, map_size=1048576*10000):
    env_out = lmdb.open(root, map_size=map_size,
                           sync=False,
                           metasync=False,
                           map_async=True,
                           writemap=True,
                           max_dbs=1)

    out_db = env_out.open_db(
        key='cpg-data'.encode('utf-8'), create=True
    )

    try:
        with env_out.begin(write=True, db=out_db) as txn2:

            for idx, data in stream:

                if isinstance(data, pb.AnnotatedTree):
                    data = data.SerializeToString()

                txn2.put(
                    ("%d" % idx).encode('utf-8'),
                    data,
                    db=out_db
                )
    finally:
        env_out.sync(True)

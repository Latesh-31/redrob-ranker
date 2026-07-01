"""Build an HNSW ANN index from saved candidate embeddings.
Saves index to embeddings/candidates_hnsw.bin
Usage:
    python scripts/build_hnsw_index.py --embeddings embeddings/candidate_embeddings.npy --ids embeddings/candidate_ids.json --out embeddings/candidates_hnsw.bin
"""
import argparse
import json
import os

import numpy as np

try:
    import hnswlib
except Exception as e:
    raise SystemExit('hnswlib is required. pip install hnswlib')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--out', default='embeddings/candidates_hnsw.bin')
    p.add_argument('--ef_construction', type=int, default=200)
    p.add_argument('--M', type=int, default=16)
    args = p.parse_args()

    vecs = np.load(args.embeddings)
    num_elements, dim = vecs.shape
    print(f'Loaded embeddings: {num_elements} x {dim}')

    with open(args.ids, 'r') as f:
        ids = json.load(f)
    if len(ids) != num_elements:
        print('Warning: ids length != embeddings length')

    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.M)
    print('Adding items...')
    index.add_items(vecs, np.arange(num_elements))
    index.set_ef(50)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    index.save_index(args.out)
    print('Saved index to', args.out)

    # save metadata
    meta = {'num_elements': int(num_elements), 'dim': int(dim)}
    with open(args.out + '.meta.json', 'w') as f:
        json.dump(meta, f)

    print('Done.')


if __name__ == '__main__':
    main()

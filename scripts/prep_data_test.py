#!/usr/bin/env python3

import polars as pl
from pathlib import Path

splits = {'train': 'train.csv', 'validation': 'valid.csv'}
src = 'hf://datasets/COGNANO/VHHCorpus-2M/' + splits['validation']

out_dir = Path('data/processed').resolve()
out_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(src)

df2 = (
    df
    .with_columns([
        pl.col('VHH_sequence').alias('sequence'),
        pl.col('VHH_sequence').str.count_matches('C').alias('n_cys'),
        pl.col('VHH_sequence').str.count_matches('N[^P][TS]').alias('n_glycosilations'),
    ])
    .with_columns(
        # suitable_for_expression: True if no glycosylations AND exactly 2 cysteines
        ( (pl.col('n_glycosilations') == 0) & (pl.col('n_cys') == 2) ).cast(pl.Int8).alias('target')
    )
    .select(['sequence', 'target'])
)

df2.write_csv(out_dir / 'test_vhh_41K.tsv', separator='\t')
df2.sample(n=200, seed=0).write_csv(out_dir / 'test_vhh_200.tsv', separator='\t')
df2.sample(n=2000, seed=0).write_csv(out_dir / 'test_vhh_2K.tsv', separator='\t')
df2.sample(n=20000, seed=0).write_csv(out_dir / 'test_vhh_20K.tsv', separator='\t')

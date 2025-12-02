import os
import argparse
import pandas as pd
from loguru import logger


def get_args():
    parser = argparse.ArgumentParser(description='Protein fitnes modeling via protein language models')
    parser.add_argument('-i', '--input_path', type=str, help='Input data path, examples/Savinase_test/data.csv')
    parser.add_argument('-c', '--seq_col', type=str, default='seq', help='Sequence column name, default=seq')
    return parser.parse_args()


def main():
    args = get_args()
    data_file = args.input_path

    # check if data file exists
    if not os.path.exists(data_file):
        logger.error(f'file {data_file} does not exist')
        return

    logger.info(f'reading {args.seq_col} from file {data_file}')
    df_data = pd.read_csv(data_file)

    with open(data_file + '.fasta', 'w') as f:
        for i, seq in enumerate(df_data[args.seq_col]):
            f.write(f'>seq_{i}\n{seq}\n')

    logger.info(f'fasta file saved to path: {data_file}.fasta')


if __name__ == "__main__":
    main()

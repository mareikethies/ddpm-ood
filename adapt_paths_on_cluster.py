import os
import argparse
import pandas as pd
from pathlib import Path

# just for testing
# os.environ['data_root'] = '/media/mareike/Elements/Data/3DMedicalDecathlon'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_data_root', type=str)
    args = parser.parse_args()

    new_data_dir = Path(args.new_data_root)
    if not (new_data_dir / 'data_splits').exists():
        (new_data_dir / 'data_splits').mkdir(exist_ok=True, parents=True)

    base_folder = Path(os.getenv('data_root')) / 'data_splits'

    for file in base_folder.iterdir():
        contents = pd.read_csv(file)
        file_name = file.name
        dict = {}
        for i, path in enumerate(contents):
            path = Path(path)
            new_path = new_data_dir / path.parts[-4] / path.parts[-3] / path.parts[-2] / path.name
            dict.update({str(path): str(new_path)})
        contents.rename(columns=dict, inplace=True)
        contents.to_csv(new_data_dir / 'data_splits' / file_name)


if __name__ == '__main__':
    main()

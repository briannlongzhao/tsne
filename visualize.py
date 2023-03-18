import numpy as np
from PIL import Image
import argparse
import os
from glob import glob
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

from model import FeatureExtractor



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dirs', type=str, nargs='+', help="root directory that contains images of each class")
    parser.add_argument('--img_json', type=str, default=None, help="json that includes root directory that contains images of each class")
    parser.add_argument('--save_dir', type=str, default="output/", help="where to save visualization")
    parser.add_argument('--feat_extractor', type=str, default="xception", help="feature extactor model to use")
    args = parser.parse_args()
    return args


def get_embs_and_labels(dir_list, categories):
    featex = FeatureExtractor(args.feat_extractor, batch_size=32)
    emb_list, label_list = [], []
    for img_dir in dir_list:
        cat_name = os.path.basename(os.path.normpath(img_dir))
        img_list = []
        for img_path in tqdm(glob(os.path.join(img_dir, '**/*.*'), recursive=True), desc=f"loading {cat_name}"):
            img = np.asarray(Image.open(img_path))
            img_list.append(img)
        print("extracting image features")
        embs = featex.compute_embedding(img_list)
        emb_list += [item for item in embs]
        label_list += len(embs)*[categories.index(cat_name)]
    assert len(emb_list) == len(label_list), f"{len(emb_list)} {len(label_list)}"
    return emb_list, label_list


def plot_tsne(embs, labels, categories, save_dir):
    x = TSNE(n_components=2).fit_transform(embs)
    df = pd.DataFrame(x, columns=["x1","x2"])
    df['y'] = [categories[y] for y in labels]
    fig = sns.scatterplot(data=df, x="x1", y="x2", hue='y').get_figure()
    plot_name = '_'.join(categories)+".png"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(os.path.join(save_dir, plot_name))


if __name__ == "__main__":
    args = parse_arguments()
    if args.img_dirs:
        cat_list = []
        for img_dir in args.img_dirs:
            cat_list.append(os.path.basename(os.path.normpath(img_dir)))
        embs, labels = get_embs_and_labels(args.img_dirs, cat_list)
        plot_tsne(embs, labels, cat_list, args.save_dir)
    elif args.img_json:
        

    
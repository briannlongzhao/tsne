import numpy as np
from PIL import Image
import argparse
import os
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
import pandas as pd
import torch
import seaborn as sns

from model import FeatureExtractor



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dirs', type=str, nargs='+', help="root directory that contains images of each class")
    parser.add_argument('--img_txt', type=str, default=None, help="json that includes root directory that contains images of each class")
    # Each line in img_txt should be:  {real_path}\t{syn_path}
    parser.add_argument('--save_dir', type=str, default="output/", help="where to save visualization")
    parser.add_argument('--feat_extractor', type=str, default="xception", help="feature extactor model to use")
    args = parser.parse_args()
    return args


def get_embs_and_labels(dir_list, categories):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    featex = FeatureExtractor(args.feat_extractor, batch_size=64).to(device)
    emb_list, label_list = [], []
    for img_dir in dir_list:
        if "syn" in categories and "real" in categories:
            cat_name = "real" if "ILSVRC2012" in img_dir else "syn"
        else:
            cat_name = os.path.basename(os.path.normpath(img_dir))
        img_list = []
        assert os.path.exists(img_dir), img_dir
        for img_path in tqdm(glob(os.path.join(img_dir, '**/*.*'), recursive=True), desc=f"loading {cat_name}"):
            img = np.asarray(Image.open(img_path))
            img_list.append(img)
        print("extracting image features")
        embs = featex.compute_embedding(img_list)
        emb_list += [item for item in embs]
        label_list += len(embs)*[categories.index(cat_name)]
    assert len(emb_list) == len(label_list), f"{len(emb_list)} {len(label_list)}"
    return emb_list, label_list


def plot_tsne(embs, labels, categories, save_dir, title=None):
    x = TSNE(n_components=2).fit_transform(embs)
    df = pd.DataFrame(x, columns=["x1","x2"])
    df['y'] = [categories[y] for y in labels]
    assert len(df['y']) == len(df["x1"])
    fig = sns.scatterplot(data=df, x="x1", y="x2", hue='y')
    if title is not None:
        fig.set_title(title)
    fig = fig.get_figure()
    plot_name = title if title is not None else '_'.join(categories)+".png"
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
    elif args.img_txt:
        f = open(args.img_txt)
        for line in f.readlines():
            dir_list = line.replace('\n','').split('\t')
            category = os.path.basename(os.path.normpath(dir_list[-1]))
            embs, labels = get_embs_and_labels(dir_list, ["real","syn"])
            plot_tsne(embs, labels, ["real","syn"], args.save_dir, title=category)
        f.close()
    else:
        raise NotImplementedError

    
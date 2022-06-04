import numpy as np
import argparse
from tqdm import tqdm

from graphgallery.gallery import callbacks
import graphgallery as gg

gg.set_backend("th")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="cora",
                    help="Datasets. (default: cora)")
parser.add_argument("--backbone", nargs="?", default="SSGC",
                    help="GCN, SGC or SSGC. (default: SSGC)")
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for model and dataset. (default: 42)')
parser.add_argument('--K', type=int, default=2,
                    help='Propagation step for SGC or SSGC (default: 2)')
parser.add_argument('--k', type=int, default=30,
                    help='approximation rank (default: 30)')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--runs', type=int, default=10,
                    help='Number of runs. (default: 10)')
args = parser.parse_args()

dataset = args.dataset
data = gg.datasets.NPZDataset(
    dataset, root='~/GraphData/datasets/', transform="standardize",  verbose=False)
splits = data.split_nodes(random_state=15)
graph = data.graph

attacker_name = 'Nettack'
direct_attack = True
adv_edges = np.load(
    f'adversarial_edges/targeted/{dataset}_{attacker_name}{"_In" if not direct_attack else ""}.npy', allow_pickle=True).item()
targets = np.array(list(adv_edges.keys()))




for runs in range(args.runs):
    if args.backbone == 'GCN':
        trainer = gg.gallery.nodeclas.SATSGC(
            device='cuda', seed=args.seed+runs, lr=args.lr).setup_graph(graph, k=args.k).build()  # cora_ml
    if args.backbone == 'SGC':
        trainer = gg.gallery.nodeclas.SATGCN(
            device='cuda', seed=args.seed+runs, lr=args.lr).setup_graph(graph, k=args.k).build()  # cora_ml
    if args.backbone == 'SSGC':
        if args.dataset == 'pubmed':
            trainer = gg.gallery.nodeclas.SATSSGC(
                device='cuda', seed=args.seed+runs, lr=args.lr).setup_graph(graph, k=args.k).build(K=5, alpha=0.2, hids=16, acts='relu') 
        else:
            trainer = gg.gallery.nodeclas.SATSSGC(
                device='cuda', seed=args.seed+runs, lr=args.lr).setup_graph(graph, k=args.k).build(K=5, alpha=0.2) 

    cb = callbacks.ModelCheckpoint('model.pth', monitor='val_accuracy')
    trainer.fit(splits.train_nodes, splits.val_nodes,
                verbose=1, epochs=args.epochs, callbacks=[cb])
    results = trainer.evaluate(targets)
    print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')

    pbar = tqdm(targets)
    degree = graph.d.astype('int')
    num = 0

    for target in pbar:
        target_label = graph.y[target]
        modified_graph = graph.from_flips(
            edge_flips=np.transpose(adv_edges[target][:degree[target]]))
        trainer.setup_graph(modified_graph, adj_transform=None)
        p = trainer.predict(target)
        if p.argmax() == target_label:
            num += 1
        acc = num / len(targets)
        pbar.set_description(f"Accuracy {acc:.2%}")

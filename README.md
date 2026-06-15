# Semi-Supervised Graph Neural Network for Network Traffic Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Author**: Aron Wang (WYN)  
> **Email**: [wyn@mails.guet.edu.cn](mailto:wyn@mails.guet.edu.cn)  
> **Affiliation**: Guilin University of Electronic Technology, AI Undergraduate

This repository implements a **semi‑supervised graph neural network** for flow‑level network traffic classification. It builds dynamic time‑windowed graphs (IPs as nodes, flows as edges), applies **GCNs** for spatial learning, **LSTMs** for temporal modeling, and uses **pseudo‑labeling** to leverage unlabeled edges.

---

## 📌 Key Features

- **Dynamic graph construction** per minute from raw flow logs (supports 2012, 2017, 2020 datasets)
- **Semi‑supervised learning** – only 20% of edges are labeled; the rest use pseudo‑labels (confidence threshold 0.8)
- **ROEN model** – Recurrent Graph Neural Network with Edge‑wise features (GCN + LSTM + MLP)
- **Model ensemble** – for multi‑file datasets (e.g., 2012), trains separate models and averages predictions
- **Evaluation** – Accuracy, Precision, Recall, F1, AUC (only on truly labeled edges)

---

## 📂 Project Structure
.
├── network.py # ROEN model definition
├── untils.py # data loading, graph construction, dataset utilities
├── train_2012.py # full pipeline for 2012 dataset (multi‑file, ensemble)
├── train_2017.py # training for 2017 dataset
├── train_2020.py # training for 2020 dataset
├── test.py # quick training using pre‑saved graph data (local path)
└── init.py # package initializer

text

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Aron_Wangnaaa/Public.git
cd Public
2. Install dependencies
bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version
pip install torch_geometric pandas numpy scikit-learn matplotlib tqdm
3. Prepare data
Place your CSV files in the expected directories:

For train_2017.py → data/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv

For train_2020.py → data/Darknet/Darknet.csv

For train_2012.py → DATA/2012/Testbed*Flows.csv (multiple files)

Or modify the paths in the scripts.

4. Run training
2012 dataset (ensemble)

bash
python train_2012.py
2017 dataset

bash
python train_2017.py
2020 dataset

bash
python train_2020.py
Using pre‑built graph data (change graph_data_dir in test.py)

bash
python test.py
🧠 Model Architecture (ROEN)
The ROEN model processes a sequence of graphs (one per time window):

Node features → MLP → multi‑layer GCN

Edge features → MLP → multi‑layer Linear

Temporal modeling → LSTM for node features + LSTM for edge features

Edge classification – concatenates source node, target node, and edge LSTM outputs → final MLP → class logits

During training, pseudo‑labels are generated for unlabeled edges whose softmax probability exceeds pseudo_label_threshold (default 0.8).

📊 Evaluation
The evaluate() function computes metrics only on the truly labeled edges (as defined by labeled_mask). This reflects realistic semi‑supervised evaluation.

Outputs include:

Accuracy (%), Precision (%), Recall, F1, AUC

Confusion matrix

Training curves (loss, pseudo‑label counts, validation metrics)

Detailed predictions (detailed_results.csv)

📁 Outputs
All models and results are saved under models/:

models/2012/group1_model_best.pth, group2_model_best.pth, ensemble_results.csv

models/2017/model_epoch_*.pth

models/2020/model_epoch_*.pth

Training plots: training_loss.png, validation_metrics.png, overfitting_detection.png

⚙️ Customization
Adjust labeled ratio
In create_graph_data_XXXX(), change labeled_ratio=0.2 to any value (e.g., 0.1 for 10% labeled edges).

Change pseudo‑label threshold
Inside network.py, modify self.pseudo_label_threshold = 0.8.

Different dataset
Implement your own create_ip_mapping_YOURDATA() and create_graph_data_YOURDATA() in untils.py – ensure the returned Data object contains:

x, edge_index, edge_attr

edge_labels (‑1 for unlabeled)

labeled_mask (bool tensor)

📝 Notes
The 2012 training script splits files into two groups and trains separate models; you can adjust grouping in train_2012.py.

For small datasets (<5 graphs), test.py automatically switches to leave‑one‑out or uses all data for training.

All tensors are moved to GPU automatically if available (CUDA).

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

📧 Contact
Wang Yunan (Aron) – wyn@mails.guet.edu.cn
Feel free to reach out for questions or collaboration.

Last updated: June 2026

text

You can copy the above content into a file named `README.md` and place it in your repository root. It’s ready for GitHub.

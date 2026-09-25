import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from cnn import VGG16, ResNet18
from ctsnet.model import *
from ctsnet.work.CTSNet_firstorder import CTSNet_firstorder
from ctsnet.work.CTSNet_nocrossscale import CTSNet_nocrossscale
from ctsnet.work.CTSNet_novit import CTSNet_novit

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def txt_to_raw_feats(txt_path, model, device, transform):
    """读取txt，返回 (feats: NxD, labels: N,)"""
    model.eval()
    feats_list = []
    labels_list = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_p, cls_id_str = line.split()
            cls_id = int(cls_id_str)
            img = Image.open(img_p).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.get_embedding(img_tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            feats_list.append(feat.squeeze(0))
            labels_list.append(cls_id)
    feats = torch.stack(feats_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    return feats, labels

def raw_to_grouped_features(gallery_feats, gallery_labels, probe_feats, probe_labels):
    """
    gallery + probe 合并，按label分组；
    返回：
        class_ids: list，类别ID
        grouped: Tensor (C, S, D)，C类别数，S每类样本数（对齐后）
    """
    all_feats = torch.cat([gallery_feats, probe_feats], dim=0)
    all_labels = torch.cat([gallery_labels, probe_labels], dim=0)

    cls2feats = {}
    for feat, lab in zip(all_feats, all_labels):
        lab = int(lab.item())
        if lab not in cls2feats:
            cls2feats[lab] = []
        cls2feats[lab].append(feat)

    class_ids = sorted(cls2feats.keys())
    # 取每类最少样本做对齐
    min_sample = min(len(v) for v in cls2feats.values())
    grouped_list = []
    for cid in class_ids:
        arr = torch.stack(cls2feats[cid][:min_sample], dim=0)  # S x D
        grouped_list.append(arr)
    grouped = torch.stack(grouped_list, dim=0)  # C x S x D
    return class_ids, grouped

def compute_eer(sims, labels):
    """
    sims: 一维numpy数组，相似度
    labels: 一维numpy数组，1同类，-1异类
    return eer
    """
    pos = sims[labels == 1]
    neg = sims[labels == -1]
    pos = np.sort(pos)
    neg = np.sort(neg)
    n_pos = len(pos)
    n_neg = len(neg)

    frr = []
    far = []
    for p in pos:
        frr.append(np.sum(pos <= p) / n_pos)
        far.append(np.sum(neg >= p) / n_neg)
    frr = np.array(frr)
    far = np.array(far)

    # 找FRR与FAR交点
    idx = np.argmin(np.abs(frr - far))
    eer = (frr[idx] + far[idx]) / 2.0
    return eer

def grouped_to_score_file_and_eer(grouped, score_txt_path):
    """
    grouped: C x S x D
    展开所有样本两两对，写score.txt，返回eer
    """
    C, S, D = grouped.shape
    all_feats = grouped.reshape(-1, D)  # (C*S, D)
    sim = all_feats @ all_feats.T       # N x N
    N = C * S

    # 构造标签矩阵
    cls_ids = torch.arange(C).unsqueeze(1).repeat(1, S).reshape(-1)  # N
    same_cls = (cls_ids.unsqueeze(1) == cls_ids.unsqueeze(0)).float()
    label_mat = torch.where(same_cls > 0, torch.tensor(1.0), torch.tensor(-1.0))

    # 只取上三角不含对角线（避免自匹配、重复对）
    i, j = torch.triu_indices(N, N, offset=1)
    sim_flat = sim[i, j].cpu().numpy()
    lab_flat = label_mat[i, j].cpu().numpy()

    # 写score.txt
    with open(score_txt_path, "w", encoding="utf-8") as f:
        for s, l in zip(sim_flat, lab_flat):
            f.write(f"{s:.8f} {int(l)}\n")

    eer = compute_eer(sim_flat, lab_flat)
    return eer

def cross_dataset_eval_eer(
    source_dataset, target_dataset, model_cls, model_name, num_classes,
    weights_root, data_root, gallery_name="val.txt", probe_name="test.txt",
    score_dir="./score"
):
    weight_path = os.path.join(weights_root, source_dataset, model_name, "ckpts", "best.pth")
    gallery_txt = os.path.join(data_root, target_dataset, gallery_name)
    probe_txt = os.path.join(data_root, target_dataset, probe_name)

    print(f"\n【Cross Test】Source:{source_dataset} -> Target:{target_dataset} | Model:{model_name}")
    print(f"Weight: {weight_path}")
    print(f"Gallery: {gallery_txt}")
    print(f"Probe: {probe_txt}")

    model = model_cls(num_classes).to(DEVICE)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    gallery_feats, gallery_labels = txt_to_raw_feats(gallery_txt, model, DEVICE, transform)
    probe_feats, probe_labels = txt_to_raw_feats(probe_txt, model, DEVICE, transform)

    class_ids, grouped = raw_to_grouped_features(gallery_feats, gallery_labels, probe_feats, probe_labels)
    os.makedirs(score_dir, exist_ok=True)
    score_txt_path = os.path.join(score_dir, f"{source_dataset}_{target_dataset}_{model_name}_score.txt")

    eer = grouped_to_score_file_and_eer(grouped, score_txt_path)
    print(f"EER = {eer:.4f}, score saved to {score_txt_path}")
    return {
        "source": source_dataset,
        "target": target_dataset,
        "model": model_name,
        "eer": eer
    }

if __name__ == "__main__":
    model_map = {
        # "CompNet": CompNet,
        # "sf2net": SF2Net,
        # "CTSNet": CTSNet,
        # "ccnet": ccnet,
        # "ctsnet_firstorder": CTSNet_firstorder,
        # "ctsnet_nocrossscale": CTSNet_nocrossscale,
        # "ctsnet_novit": CTSNet_novit
        "resnet18": ResNet18,
        "vgg16": VGG16
    }
    source_num_classes = {
        "CASIA": 620,
        "COEP": 163,
        "Tongji": 600,
        "IITD": 460
    }

    data_root = r"D:\PythonProject3\ctsnet\dataset\data_x_1_!"
    weights_root = r"D:\PythonProject3\ctsnet\work\cnn_weights"
    output_dir = r"D:\PythonProject3\ctsnet\work\result"
    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, "cross_eer_result_cnn.csv")

    source_list = ["CASIA", "COEP", "IITD",'Tongji']
    target_list = ["CASIA", "COEP", "Tongji", "IITD"]

    rows = []
    for src_ds in source_list:
        for tgt_ds in target_list:
            for model_name, model_cls in model_map.items():
                num_cls = source_num_classes[src_ds]
                try:
                    res = cross_dataset_eval_eer(
                        source_dataset=src_ds,
                        target_dataset=tgt_ds,
                        model_cls=model_cls,
                        model_name=model_name,
                        num_classes=num_cls,
                        weights_root=weights_root,
                        data_root=data_root,
                        score_dir="./score"
                    )
                    rows.append(res)
                except Exception as e:
                    print(f"FAIL {src_ds}->{tgt_ds} {model_name}: {e}")
                    rows.append({
                        "source": src_ds,
                        "target": tgt_ds,
                        "model": model_name,
                        "eer": None
                    })

    fieldnames = ["source", "target", "model", "eer"]
    with open(csv_out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll done. EER result saved to {csv_out_path}")
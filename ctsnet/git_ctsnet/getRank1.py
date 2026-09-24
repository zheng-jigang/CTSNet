# txt_to_matrix:path1.txt,path2.txt->gallery_matrix,probe_matrix
# get_rank:gallery_matrix,probe_matrix->rank1
# txt_structure:
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0003\0003_0004.bmp 2
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0003\0003_0005.bmp 2
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0003\0003_0006.bmp 2
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0004\0004_0004.bmp 3
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0004\0004_0005.bmp 3
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0005\0005_0004.bmp 4
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0005\0005_0005.bmp 4
# D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0005\0005_0006.bmp 4
import os
import csv
import torch
from PIL import Image
from torchvision import transforms

from cnn import ResNet18, VGG16
from compnew_convmix import compnew
from ctsnet.model import *
from ctsnet.work.CTSNet_firstorder import CTSNet_firstorder
from ctsnet.work.CTSNet_nocrossscale import CTSNet_nocrossscale
from ctsnet.work.CTSNet_novit import CTSNet_novit

22
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def txt_to_matrix(txt_path, model, device, transform):
    """
    读txt（img_path class_id），批量提取L2归一化特征
    return feats(N,D), labels(N,)
    """
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


def get_rank_vectorized(gallery_feats, gallery_labels, probe_feats, probe_labels):
    """
    完全向量化计算Rank-1，无probe内for循环
    gallery_feats: G × D, L2归一化
    gallery_labels: (G,)
    probe_feats: P × D, L2归一化
    probe_labels: (P,)
    return rank1_acc
    """
    # P×G 余弦相似度矩阵
    sim = probe_feats @ gallery_feats.T
    # 每行按相似度降序，得到gallery索引 P×G
    _, sort_idx = torch.sort(sim, dim=1, descending=True)
    # P×G：按相似度排序后的gallery标签
    sorted_g_labels = gallery_labels[sort_idx]
    # 取每个probe最相似的gallery标签（第一列）
    top1_label = sorted_g_labels[:, 0]
    # 与probe真实标签对比
    hit = (top1_label == probe_labels).float()
    rank1_acc = hit.mean().item()
    return rank1_acc


def cross_dataset_eval(
    source_dataset, target_dataset, model_cls, model_name, num_classes,
    weights_root, data_root, gallery_name="val.txt", probe_name="test.txt"
):
    """
    source_dataset: 训练集名称（权重所属）
    target_dataset: 测试集名称（gallery/probe所在）
    model_cls: 模型构造类/函数
    model_name: 字符串标识，用于找权重目录
    """
    weight_path = os.path.join(weights_root, source_dataset, model_name, "ckpts", "best.pth")
    gallery_txt = os.path.join(data_root, target_dataset, gallery_name)
    probe_txt = os.path.join(data_root, target_dataset, probe_name)

    print(f"\n【Cross Test】Source:{source_dataset} -> Target:{target_dataset} | Model:{model_name}")
    print(f"Weight: {weight_path}")
    print(f"Gallery: {gallery_txt}")
    print(f"Probe: {probe_txt}")

    # build & load model
    model = model_cls(num_classes).to(DEVICE)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])

    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    gallery_feats, gallery_labels = txt_to_matrix(gallery_txt, model, DEVICE, transform)
    probe_feats, probe_labels = txt_to_matrix(probe_txt, model, DEVICE, transform)

    rank1 = get_rank_vectorized(gallery_feats, gallery_labels, probe_feats, probe_labels)
    print(f"Rank-1 = {rank1:.4f}")
    return {
        "source": source_dataset,
        "target": target_dataset,
        "model": model_name,
        "rank1": rank1
    }


if __name__ == "__main__":
    # 映射：模型字符串名 -> 构造器
    model_map = {
        # "CompNet": CompNet,
        # "sf2net": SF2Net,
        # "CTSNet": CTSNet,
        # "ccnet": ccnet,
        #
        # "resnet18":ResNet18,
        # "vgg16":VGG16
        "compnew":compnew
    }
    # 你需要维护 source数据集对应的 num_classes（训练时用的类别数）
    source_num_classes = {
        "CASIA": 620,
        "COEP": 163,
        "Tongji": 600,
        "IITD": 460
    }

    data_root = r"D:\PythonProject3\ctsnet\dataset\data_x_1_!"
    weights_root = r"D:\PythonProject3\ctsnet\compnew_weights_conmix"
    output_dir = r"D:\PythonProject3\ctsnet\work\result"
    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, "cross_rank1_result_compnew_mixconv.csv")

    source_list = ["Tongji"]
    target_list = ["CASIA","COEP","Tongji","IITD"]

    rows = []
    for src_ds in source_list:
        for tgt_ds in target_list:
            for model_name, model_cls in model_map.items():
                num_cls = source_num_classes[src_ds]
                try:
                    res = cross_dataset_eval(
                        source_dataset=src_ds,
                        target_dataset=tgt_ds,
                        model_cls=model_cls,
                        model_name=model_name,
                        num_classes=num_cls,
                        weights_root=weights_root,
                        data_root=data_root,
                        gallery_name="val.txt",
                        probe_name="test.txt"
                    )
                    rows.append(res)
                except Exception as e:
                    print(f"FAIL {src_ds}->{tgt_ds} {model_name}: {e}")
                    rows.append({
                        "source": src_ds,
                        "target": tgt_ds,
                        "model": model_name,
                        "rank1": None
                    })

    # 写出CSV
    fieldnames = ["source","target","model","rank1"]
    with open(csv_out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll done. Result saved to {csv_out_path}")
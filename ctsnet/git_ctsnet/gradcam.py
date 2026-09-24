import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from CTSNet import CTSNet


def load_single_image(img_path, target_size=(128, 128), device="cuda"):
    img = cv2.imread(img_path, 0)
    if img is None:
        raise FileNotFoundError(f"无法读取图片 {img_path}")
    img = cv2.resize(img, target_size, cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor, img


class SingleModelVisualizer:
    """cb1注册maxpool，cb2/cb3注册conv2_2"""
    def __init__(self, model, block_names):
        self.model = model
        self.block_names = block_names
        self.hooks = dict()
        self.buffer = {name: {"zeta": None, "grad": None} for name in block_names}
        self._register_hooks()

    def _register_hooks(self):
        for name in self.block_names:
            cb = getattr(self.model, name)
            buf = self.buffer[name]

            def forward_hook(storage):
                def hook(mod, inp, out):
                    storage["zeta"] = out.detach()
                return hook

            def backward_hook(storage):
                def hook(mod, grad_in, grad_out):
                    storage["grad"] = grad_out[0]
                return hook

            if name == "cb1":
                target_layer = cb.maxpool
            else:
                if not hasattr(cb, "conv2_2"):
                    raise AttributeError(f"{name} 不存在 conv2_2！")
                target_layer = cb.conv2_2

            hf = target_layer.register_forward_hook(forward_hook(buf))
            hb = target_layer.register_full_backward_hook(backward_hook(buf))
            self.hooks[name] = (hf, hb)
            print(f"[Hook Registered] {name} -> {target_layer}")

    def remove_hooks(self):
        for h1, h2 in self.hooks.values():
            h1.remove()
            h2.remove()


def build_origin_overlay_pair(gray_img, cam_map, alpha=0.5):
    """仅输出原图rgb 和 gradcam叠加图，不再包含特征图"""
    img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heat, alpha, 0)
    return img_rgb, overlay


def build_single_module_image_only_cam(gray_img, data, block_name):
    """单模块：标签栏 + [原图｜Grad‑CAM叠加图]，去掉特征图"""
    origin_rgb, overlay_img = build_origin_overlay_pair(gray_img, data["cam_map"])
    row_img = np.hstack([origin_rgb, overlay_img])

    label = np.zeros((row_img.shape[0], 60, 3), dtype=np.uint8)
    cv2.putText(label, block_name.upper(), (5, row_img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row_img = np.hstack([label, row_img])

    title_bar = np.zeros((40, row_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, block_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    single_out = np.vstack([title_bar, row_img])
    return single_out, origin_rgb, overlay_img


def build_full_image(result, raw_gray, tag):
    all_rows = []
    for blk in blocks:
        data = result[blk]
        origin_rgb, overlay_img = build_origin_overlay_pair(raw_gray, data["cam_map"])
        row_img = np.hstack([origin_rgb, overlay_img])
        label = np.zeros((row_img.shape[0], 60, 3), dtype=np.uint8)
        cv2.putText(label, blk.upper(), (5, row_img.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        row_img = np.hstack([label, row_img])
        all_rows.append(row_img)
    full_pic = np.vstack(all_rows)
    title_bar = np.zeros((40, full_pic.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    full_pic = np.vstack([title_bar, full_pic])
    return full_pic


if __name__ == "__main__":
    # ==================== 配置区 ====================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WEIGHT_PATH = r"D:\PythonProject3\ctsnet\weights\Tongji\ctsnet\ckpts\best.pth"

    IMG_A_PATH = r"D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0011\0011_0005.bmp"
    IMG_B_PATH = r"D:\palmprintrecognition\dataset\IITD_dataset_official_renamed\0041\0041_0002.bmp"
    ROOT_SAVE_DIR = r"./gradcam_result"
    # =================================================

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(ROOT_SAVE_DIR, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"本次实验保存目录：{exp_dir}")

    subA_dir = os.path.join(exp_dir, "imageA")
    subB_dir = os.path.join(exp_dir, "imageB")
    os.makedirs(subA_dir, exist_ok=True)
    os.makedirs(subB_dir, exist_ok=True)


    def load_model(model, weight_path):
        ckpt = torch.load(weight_path, map_location=DEVICE)
        if "model" in ckpt:
            weights = ckpt["model"]
        elif "state_dict" in ckpt:
            weights = ckpt["state_dict"]
        else:
            weights = ckpt
        model.load_state_dict(weights, strict=True)
        model.to(DEVICE)
        model.eval()
        return model


    modelA = CTSNet(num_classes=600).to(DEVICE)
    modelB = CTSNet(num_classes=600).to(DEVICE)
    load_model(modelA, WEIGHT_PATH)
    load_model(modelB, WEIGHT_PATH)

    blocks = ["cb1", "cb2", "cb3"]
    visA = SingleModelVisualizer(modelA, blocks)
    visB = SingleModelVisualizer(modelB, blocks)

    imgA_tensor, rawA_gray = load_single_image(IMG_A_PATH, device=DEVICE)
    imgB_tensor, rawB_gray = load_single_image(IMG_B_PATH, device=DEVICE)

    imgA_tensor.requires_grad = True
    imgB_tensor.requires_grad = True

    logitsA, embA = modelA(imgA_tensor)
    logitsB, embB = modelB(imgB_tensor)

    sim_global = F.cosine_similarity(embA, embB, dim=-1).sum()
    modelA.zero_grad()
    modelB.zero_grad()
    sim_global.backward()

    H_in, W_in = imgA_tensor.shape[-2], imgA_tensor.shape[-1]

    def collect_result(visualizer):
        res = {}
        feat_dict = {}
        for blk in blocks:
            buf = visualizer.buffer[blk]
            zeta = buf["zeta"][0]
            grad = buf["grad"][0]

            weight = torch.mean(grad, dim=(1, 2), keepdim=True)
            cam = torch.sum(weight * zeta, dim=0)
            cam = torch.relu(cam)
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)
            cam_np = cam.cpu().numpy()
            cam_resize = cv2.resize(cam_np, (W_in, H_in))

            feat_mean = torch.mean(zeta, dim=0)
            feat_mean = feat_mean - torch.min(feat_mean)
            feat_mean = feat_mean / (torch.max(feat_mean) + 1e-8)
            feat_np = feat_mean.cpu().numpy()
            feat_resize = cv2.resize(feat_np, (W_in, H_in))

            res[blk] = {
                "zeta_tensor": zeta,
                "cam_map": cam_resize,
                "conv_feature": feat_resize,
                "conv_raw_tensor": zeta
            }
            feat_flat = F.normalize(zeta.flatten(), p=2, dim=0)
            feat_dict[blk] = feat_flat
        return res, feat_dict

    resA, featA = collect_result(visA)
    resB, featB = collect_result(visB)

    sim_cb1 = torch.dot(featA["cb1"], featB["cb1"]).item()
    sim_cb2 = torch.dot(featA["cb2"], featB["cb2"]).item()
    sim_cb3 = torch.dot(featA["cb3"], featB["cb3"]).item()
    sim_global_val = sim_global.item()

    print("\n===== 相似度汇总 =====")
    print(f"cb1 conv2_2: {sim_cb1:.6f}")
    print(f"cb2 conv2: {sim_cb2:.6f}")
    print(f"cb3 conv2: {sim_cb3:.6f}")
    print(f"global emb: {sim_global_val:.6f}")

    sim_log = os.path.join(exp_dir, "similarity_log.txt")
    with open(sim_log, "w", encoding="utf-8") as f:
        f.write(f"cb1_conv2_2={sim_cb1:.6f}\n")
        f.write(f"cb2_conv2={sim_cb2:.6f}\n")
        f.write(f"cb3_conv2={sim_cb3:.6f}\n")
        f.write(f"global_emb={sim_global_val:.6f}\n")

    # ========= 保存：拼接图、独立原图、独立Grad‑CAM叠加图；不再输出特征可视化 =========
    # Image A
    for blk in blocks:
        data = resA[blk]
        single_img, origin_rgb, overlay_img = build_single_module_image_only_cam(rawA_gray, data, blk)
        cv2.imwrite(os.path.join(subA_dir, f"{blk}_pair.png"), single_img)
        cv2.imwrite(os.path.join(subA_dir, f"{blk}_origin.png"), origin_rgb)
        cv2.imwrite(os.path.join(subA_dir, f"{blk}_gradcam_overlay.png"), overlay_img)

        np.savetxt(os.path.join(subA_dir, f"{blk}_gradcam.txt"), data["cam_map"], fmt="%.6f")

    # Image B
    for blk in blocks:
        data = resB[blk]
        single_img, origin_rgb, overlay_img = build_single_module_image_only_cam(rawB_gray, data, blk)
        cv2.imwrite(os.path.join(subB_dir, f"{blk}_pair.png"), single_img)
        cv2.imwrite(os.path.join(subB_dir, f"{blk}_origin.png"), origin_rgb)
        cv2.imwrite(os.path.join(subB_dir, f"{blk}_gradcam_overlay.png"), overlay_img)

        np.savetxt(os.path.join(subB_dir, f"{blk}_gradcam.txt"), data["cam_map"], fmt="%.6f")

    # 总拼接大图保留
    fullA = build_full_image(resA, rawA_gray, "Image A")
    fullB = build_full_image(resB, rawB_gray, "Image B")
    pair_total = np.vstack([fullA, fullB])
    pair_save_path = os.path.join(exp_dir, "pair_all_visual.png")
    cv2.imwrite(pair_save_path, pair_total)
    print(f"\n合并总图已保存：{pair_save_path}")

    plt.figure(figsize=(16,14))
    plt.imshow(cv2.cvtColor(pair_total, cv2.COLOR_BGR2RGB))
    plt.title("Image A & Image B | CB1(maxpool) / CB2/CB3(conv2_2): Raw | Similarity‑Grad‑CAM Overlay")
    plt.axis("off")
    plt.show()

    visA.remove_hooks()
    visB.remove_hooks()
    print("全部可视化完成，钩子已释放！")
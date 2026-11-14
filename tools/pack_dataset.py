#!/usr/bin/env python3
"""Empaqueta imagenes + mascaras CVAT (Segmentation mask 1.1) en un zip portable.

Uso:
  python pack_dataset.py --images DIR_IMAGENES --cvat-zip masks_1.zip --out segpack_v1.zip

Salida (zip, sin compresion de JPEG):
  images/<base>.JPG   masks/<base>.png   labelmap.txt   classes.json
  manifest.csv (hash, tamano, EXIF/XMP DJI)   class_stats.csv   REPORT.txt
Las mascaras de SegmentationClass se emparejan por nombre base con las imagenes.
"""
import argparse, csv, hashlib, io, json, os, re, sys, zipfile
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

XMP_KEYS = ["GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "RelativeAltitude",
            "GimbalYawDegree", "GimbalPitchDegree", "GimbalRollDegree",
            "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_xmp(path):
    with open(path, "rb") as f:
        head = f.read(262144)
    out = {}
    for k in XMP_KEYS:
        m = re.search(rb"drone-dji:%s\s*=\s*\"([^\"]*)\"" % k.encode(), head)
        out[k] = m.group(1).decode() if m else ""
    return out

def parse_labelmap(text):
    classes = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, color = line.split(":")[0], line.split(":")[1]
        r, g, b = (int(x) for x in color.split(","))
        classes.append({"index": len(classes), "name": name, "rgb": [r, g, b]})
    return classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--cvat-zip", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cz = zipfile.ZipFile(a.cvat_zip)
    labelmap_txt = cz.read("labelmap.txt").decode("utf-8")
    classes = parse_labelmap(labelmap_txt)
    color2idx = {tuple(c["rgb"]): c["index"] for c in classes}
    masks = {}
    for n in cz.namelist():
        if n.startswith("SegmentationClass/") and n.lower().endswith(".png"):
            masks[os.path.splitext(os.path.basename(n))[0]] = n
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(a.images)
            if f.lower().endswith((".jpg", ".jpeg"))}
    pairs = sorted(set(masks) & set(imgs))
    report = [f"mascaras CVAT: {len(masks)}", f"imagenes en carpeta: {len(imgs)}",
              f"parejas: {len(pairs)}",
              f"mascaras sin imagen: {sorted(set(masks)-set(imgs))}",
              f"imagenes sin mascara (excluidas): {len(set(imgs)-set(masks))}"]
    manifest, stats = [], []
    problems = []
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as zo:
        zo.writestr("labelmap.txt", labelmap_txt)
        zo.writestr("classes.json", json.dumps(classes, indent=1))
        for i, base in enumerate(pairs, 1):
            ip = os.path.join(a.images, imgs[base])
            mb = cz.read(masks[base])
            mimg = Image.open(io.BytesIO(mb)).convert("RGB")
            marr = np.array(mimg)
            with Image.open(ip) as im:
                w, h = im.size
            ok_size = (mimg.size == (w, h))
            flat = marr.reshape(-1, 3)
            codes = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2].astype(np.int32)
            ucodes, cnt = np.unique(codes, return_counts=True)
            uniq = np.stack([(ucodes >> 16) & 255, (ucodes >> 8) & 255, ucodes & 255], axis=1)
            unknown = [tuple(u) for u in uniq if tuple(u) not in color2idx]
            if not ok_size: problems.append(f"{base}: tamano mascara {mimg.size} != imagen {(w,h)}")
            if unknown: problems.append(f"{base}: colores fuera del labelmap {unknown[:5]} ({sum(int(c) for u,c in zip(uniq,cnt) if tuple(u) in unknown)} px)")
            row = {"name": base, "image": f"images/{imgs[base]}", "mask": f"masks/{base}.png",
                   "width": w, "height": h, "mask_size_ok": ok_size,
                   "bytes": os.path.getsize(ip), "sha256": sha256(ip)}
            row.update(read_xmp(ip))
            manifest.append(row)
            srow = {"name": base}
            tot = flat.shape[0]
            for c in classes:
                srow[c["name"]] = 0
            for u, c_ in zip(uniq, cnt):
                k = color2idx.get(tuple(u))
                if k is not None:
                    srow[classes[k]["name"]] = int(c_)
            srow["unknown_px"] = int(sum(int(c_) for u, c_ in zip(uniq, cnt) if tuple(u) not in color2idx))
            stats.append(srow)
            zo.write(ip, f"images/{imgs[base]}")
            zo.writestr(f"masks/{base}.png", mb)
            if i % 10 == 0: print(f"{i}/{len(pairs)}", flush=True)
        def csv_bytes(rows):
            s = io.StringIO(); w_ = csv.DictWriter(s, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
            return s.getvalue().encode("utf-8")
        zo.writestr("manifest.csv", csv_bytes(manifest))
        zo.writestr("class_stats.csv", csv_bytes(stats))
        report.append("problemas: " + ("ninguno" if not problems else ""))
        report += problems
        zo.writestr("REPORT.txt", "\n".join(report))
    for d, rows in (("manifest.csv", manifest), ("class_stats.csv", stats)):
        with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), d), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "REPORT.txt"), "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))

if __name__ == "__main__":
    main()

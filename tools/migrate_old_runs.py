"""
Copia os checkpoints do layout antigo para o novo (não apaga nada):

  runs/resnet50/resnet50_best_model_cv1_split1.pth
      -> runs/resnet50_rgb/cv1_split1/best.pth
  runs/resnet50_hadamard/resnet50_hadamard_best_model_cv1_split1.pth
      -> runs/resnet50_hadamard/cv1_split1/best.pth
  runs/resnet50_mask/resnet50_mask_best_model_cv1_split1.pth
      -> runs/resnet50_mask/cv1_split1/best.pth

    python tools/migrate_old_runs.py            # mostra o que faria
    python tools/migrate_old_runs.py --apply    # copia
"""
import argparse
import glob
import os
import re
import shutil

# pasta antiga -> (prefixo do arquivo, experimento novo)
MAPPING = {
    "resnet50": ("resnet50_best_model", "resnet50_rgb"),
    "resnet50_hadamard": ("resnet50_hadamard_best_model", "resnet50_hadamard"),
    "resnet50_mask": ("resnet50_mask_best_model", "resnet50_mask"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    n = 0
    for old_dir, (prefix, new_exp) in MAPPING.items():
        src_dir = os.path.join(args.runs, old_dir)
        for src in sorted(glob.glob(os.path.join(src_dir, f"{prefix}_cv*_split*.pth"))):
            m = re.search(r"_(cv\d+_split\d+)\.pth$", src)
            if not m:
                continue
            dst = os.path.join(args.runs, new_exp, m.group(1), "best.pth")
            if os.path.exists(dst):
                print(f"[já existe] {dst}")
                continue
            print(f"{src}  ->  {dst}")
            n += 1
            if args.apply:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        old_log = os.path.join(src_dir, "training_log.csv")
        new_log = os.path.join(args.runs, new_exp, "training_log.csv")
        if os.path.isfile(old_log) and not os.path.exists(new_log) and old_dir != new_exp:
            print(f"{old_log}  ->  {new_log}")
            if args.apply:
                os.makedirs(os.path.dirname(new_log), exist_ok=True)
                shutil.copy2(old_log, new_log)

    print(f"\n{n} checkpoint(s) {'copiados' if args.apply else 'a copiar (use --apply)'}.")


if __name__ == "__main__":
    main()

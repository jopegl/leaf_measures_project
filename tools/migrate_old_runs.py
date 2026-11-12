"""
Reorganiza os checkpoints do layout antigo no layout novo (uma pasta por fold):

  runs/resnet50/resnet50_best_model_cv1_split1.pth
      -> runs/resnet50_rgb/cv1_split1/best.pth
  runs/resnet50_hadamard/resnet50_hadamard_best_model_cv1_split1.pth
      -> runs/resnet50_hadamard/cv1_split1/best.pth
  runs/resnet50_mask/resnet50_mask_best_model_cv1_split1.pth
      -> runs/resnet50_mask/cv1_split1/best.pth

    python tools/migrate_old_runs.py                   # só mostra o que faria (não mexe em nada)
    python tools/migrate_old_runs.py --apply           # COPIA (mantém os originais)
    python tools/migrate_old_runs.py --apply --move    # MOVE (sem duplicar; faça backup antes)

Segurança:
  - nunca sobrescreve: se o destino já existe, aquele arquivo é pulado;
  - na cópia, confere o tamanho do arquivo copiado antes de seguir;
  - no --move, os arquivos são renomeados (instantâneo no mesmo disco) e uma pasta
    antiga só é removida se ficar completamente vazia;
  - pode rodar de novo quantas vezes quiser: o que já foi feito é pulado.
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


def plan(runs: str):
    """Lista de (origem, destino) para checkpoints e logs."""
    items = []
    for old_dir, (prefix, new_exp) in MAPPING.items():
        src_dir = os.path.join(runs, old_dir)
        for src in sorted(glob.glob(os.path.join(src_dir, f"{prefix}_cv*_split*.pth"))):
            m = re.search(r"_(cv\d+_split\d+)\.pth$", src)
            if m:
                items.append((src, os.path.join(runs, new_exp, m.group(1), "best.pth")))
        old_log = os.path.join(src_dir, "training_log.csv")
        if old_dir != new_exp and os.path.isfile(old_log):
            items.append((old_log, os.path.join(runs, new_exp, "training_log.csv")))
    return items


def main():
    p = argparse.ArgumentParser(description="Reorganiza checkpoints antigos no layout novo")
    p.add_argument("--runs", default="runs")
    p.add_argument("--apply", action="store_true", help="Executa (sem isso, só mostra o plano)")
    p.add_argument("--move", action="store_true", help="Move em vez de copiar (use com --apply)")
    args = p.parse_args()

    if not os.path.isdir(args.runs):
        raise SystemExit(f"Pasta '{args.runs}' não encontrada. Rode na raiz do repo.")

    verb = "MOVER" if args.move else "COPIAR"
    done = skipped = 0
    for src, dst in plan(args.runs):
        if os.path.exists(dst):
            print(f"[já existe, pulando] {dst}")
            skipped += 1
            continue
        print(f"{src}  ->  {dst}")
        if not args.apply:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if args.move:
            os.replace(src, dst)
        else:
            tmp = dst + ".tmp"
            shutil.copy2(src, tmp)
            if os.path.getsize(tmp) != os.path.getsize(src):
                os.remove(tmp)
                raise SystemExit(f"Cópia incompleta de {src} (disco cheio?). Nada foi sobrescrito.")
            os.replace(tmp, dst)
        done += 1

    if args.apply and args.move:
        for old_dir, (_, new_exp) in MAPPING.items():
            d = os.path.join(args.runs, old_dir)
            if old_dir != new_exp and os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                print(f"Pasta antiga vazia removida: {d}")

    print()
    if args.apply:
        print(f"{done} arquivo(s) {'movidos' if args.move else 'copiados'}, {skipped} já existiam.")
    else:
        n = len(plan(args.runs)) - skipped
        print(f"Simulação: {n} arquivo(s) a {verb.lower()}, {skipped} já existem. Nada foi alterado.")
        print("Para executar: --apply (copia)  ou  --apply --move (move, sem duplicar)")


if __name__ == "__main__":
    main()
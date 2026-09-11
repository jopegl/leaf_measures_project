import os
import torch


def save_checkpoint(path: str, **kwargs):
    torch.save(kwargs, path)


def load_checkpoint_if_exists(path: str, device, model, optimizer=None, scheduler=None):
    """
    Se o checkpoint existir, carrega pesos/otimizador/scheduler e retorna
    (iteration, best_val_loss, extra_dict). Caso contrário, retorna (0, inf, {}).

    NUNCA carrega ou usa métricas de teste aqui — o checkpoint só guarda o
    estado necessário para retomar o TREINO (que é guiado por validação).
    """
    if not os.path.exists(path):
        print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
        return 0, float("inf"), {}

    print(f"Checkpoint encontrado: {path}")
    print("Carregando checkpoint...")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    iteration = checkpoint.get("iteration", 0)
    best_val_loss = checkpoint.get("val_loss", float("inf"))

    print("Checkpoint carregado!")
    print(f"Continuando da iteração: {iteration}")
    print(f"Melhor validation loss: {best_val_loss:.4f}")

    return iteration, best_val_loss, checkpoint

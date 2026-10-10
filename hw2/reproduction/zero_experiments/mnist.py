"""MLP на MNIST: инициализация по статье и обучение в Python-модуле.

Адаптация example_mnist.ipynb авторов ZerO (см. LICENSE).
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from scipy.linalg import hadamard


def ZerO_Init_on_matrix(matrix_tensor):
    """Инициализация линейного веса по Algorithm 1 статьи (версия v3)."""
    if matrix_tensor.ndim != 2:
        raise ValueError("Ожидается двумерная матрица весов")
    p, q = matrix_tensor.shape
    if p <= q:
        # Квадратный случай даёт I; прямоугольный — частичную единичную матрицу.
        return torch.eye(p, q, device=matrix_tensor.device, dtype=matrix_tensor.dtype)
    m = (p - 1).bit_length()  # Точное целое значение ceil(log2(p)).
    order = 2 ** m
    c = 2 ** (-(m - 1) / 2)  # Множитель статьи: sqrt(2 / order).
    # Срез равен произведению I*_{p,order} H_m I*_{order,q}.
    h = torch.as_tensor(hadamard(order), device=matrix_tensor.device,
                        dtype=matrix_tensor.dtype)
    return c * h[:p, :q]


def Identity_Init_on_matrix(matrix_tensor):
    """Частичная единичная матрица I* из определения 1 статьи."""
    if matrix_tensor.ndim != 2:
        raise ValueError("Ожидается двумерная матрица весов")
    p, q = matrix_tensor.shape
    # Сохраняются первые min(p, q) координат; остальные элементы равны нулю.
    return torch.eye(p, q, device=matrix_tensor.device, dtype=matrix_tensor.dtype)


class MLP(nn.Module):
    """MLP с четырьмя линейными слоями и тремя активациями ReLU."""
    def __init__(self, n_h=1024, init='ZerO'):
        super(MLP, self).__init__()
        self.init = init
        self.n_h = n_h
        self.l1 = nn.Linear(784, 784, bias=False)  
        self.l2 = nn.Linear(784, self.n_h, bias=False)  
        self.l3 = nn.Linear(self.n_h, self.n_h, bias=False)  
        self.l4 = nn.Linear(self.n_h, 10, bias=False)  

        self.apply(self._init_weights)
        
    def forward(self, x): 

        x = x.view(-1, 28 * 28)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        
        return F.log_softmax(x, dim=1)
    
    @torch.no_grad()
    def _init_weights(self, m):
        
        if self.init == 'ZerO':
            if isinstance(m, nn.Linear):
                m.weight.copy_(ZerO_Init_on_matrix(m.weight))
                
        elif self.init == 'Partial_Identity':
            if isinstance(m, nn.Linear):
                m.weight.copy_(Identity_Init_on_matrix(m.weight))
        
        elif self.init == 'Random':
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def compute_rank(tensor):
    """Число сингулярных значений выше 1e-4; это не stable rank."""
    matrix = tensor.detach().cpu().numpy()
    return int(np.linalg.matrix_rank(matrix, tol=1e-4))


class Optimizer(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr']) 

        return loss

def train(args, model, device, train_loader, optimizer, epoch, train_acc_list, train_loss_list, rank_list_dict):
    model.train()
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        loss.backward()
        optimizer.step()        
        if batch_idx % args.log_interval == 0:
            print('Эпоха: {} [{}/{} ({:.0f}%)]\tПотери: {:.6f}\tТочность: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / processed))
            
            # Записываем метрики на выбранной итерации.
            train_acc_list.append(100. * correct / processed)
            train_loss_list.append(loss.item())
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'l3' in name:
                        if name not in rank_list_dict:
                            rank_list_dict[name] = []
                            
                        # Численный ранг остаточной компоненты W − I (теорема 1).
                        identity = torch.eye(param.shape[0], device=param.device, dtype=param.dtype)
                        rank_list_dict[name].append(compute_rank(param - identity))
                
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Сумма потерь по объектам батча.
            pred = output.argmax(dim=1, keepdim=True)  # Класс с максимальной логарифмической вероятностью.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nТест: средние потери: {:.4f}, верно: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)


def train_model(model, file_dir=None, epochs=20, seed=1, data_root=None):
    # Параметры обучения исходного MNIST-примера.
    parser = argparse.ArgumentParser(description='Пример обучения MLP на MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='размер обучающего батча (по умолчанию: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='размер тестового батча (по умолчанию: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='число эпох обучения (по умолчанию: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='начальный шаг обучения (по умолчанию: 0,1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='множитель StepLR (по умолчанию: 0,7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='отключить обучение на CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='начальное значение генератора (по умолчанию: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='интервал записи метрик в батчах')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='сохранить веса обученной модели')

    parser.add_argument('--name', type=str, default='test')  
    
    parser.add_argument('--init', type=str, default='ZerO') 

    args = parser.parse_args([])
    args.epochs = epochs
    args.seed = seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(data_root or Path(__file__).resolve().parents[1]/'data'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(data_root or Path(__file__).resolve().parents[1]/'data'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = model.to(device)
    # Выводим формы параметров без печати больших матриц.
    for name, params in model.named_parameters():
        print(name, tuple(params.shape))
    optimizer = Optimizer(model.parameters(), lr=args.lr)
    
    # Списки для сохранения метрик и рангов.    
    train_acc_list = []
    train_loss_list = []
    rank_list_dict = {}
    test_history = []
    
    scheduler = StepLR(optimizer, step_size=12, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_acc_list, train_loss_list, rank_list_dict)
        accuracy = test(model, device, test_loader)
        test_history.append({"epoch": epoch, "test_accuracy": accuracy})
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), args.init + "_mnist_cnn.pt")
        
    if file_dir is not None:
        import csv
        directory = Path(file_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "test_history.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "test_accuracy"])
            writer.writeheader()
            writer.writerows(test_history)
        batches = len(train_loader)
        steps = [epoch * batches + batch + 1 for epoch in range(args.epochs)
                 for batch in range(0, batches, args.log_interval)]
        with (directory / "ranks.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "layer", "rank"])
            writer.writeheader()
            for layer, values in rank_list_dict.items():
                writer.writerows({"step": step, "layer": layer, "rank": rank}
                                 for step, rank in zip(steps, values))
        torch.save(model.state_dict(), directory / "last.pt")
    return rank_list_dict
import torch
import random

__all__ = [
  "get_grad",
  "set_grad",
  "gem_project",
  "get_weights",
  "sum_project",
  "random_project",
]


def get_grad(named_params):
  summary_grad = []
  for i, (n, p) in enumerate(named_params):
    if p.grad is not None and p.requires_grad:
      summary_grad.append(p.grad.view(-1).detach())
  summary_grad = torch.cat(summary_grad)
  return summary_grad


def get_weights(named_params):
  w = []
  for i, (n, p) in enumerate(named_params):
    if p.grad is not None and p.requires_grad:
      w.append(p.data.view(-1).detach())
  w = torch.cat(w)
  return w


def set_grad(grad, named_params):
  count = 0
  for i, (n, p) in enumerate(named_params):
    if p.grad is not None and p.requires_grad:
      s = p.grad.shape
      c = p.grad.view(-1).shape[0]
      new_grad = grad[count : (count + c)].contiguous().view(s)
      p.grad.data.copy_(new_grad)
      count += c


def gem_project(g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
  angle = (g * g_ref).sum()
  if angle < 0:

    corr = torch.dot(g, g_ref) / torch.dot(g_ref, g_ref)
    return g - corr * g_ref
  else:
    return g


def sum_project(g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
  return g + g_ref


def mean_sum_project(g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
  return (g + g_ref) / 2


def random_project(g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
  if random.random() > 0.5:
    return sum_project(g, g_ref)
  else:
    return gem_project(g, g_ref)


#     if s > self.cfg.get('agem_prob', 0.5):
#                 current_gradients = [
#                     p.grad.view(-1)
#                     for n, p in strategy.model.named_parameters() if p.requires_grad]
#                 current_gradients = torch.cat(current_gradients)
#                 angle = (current_gradients * self.reference_gradients).sum()
#                 if angle < 0:
#                     count = 0
#                     length_rep = (self.reference_gradients*self.reference_gradients).sum()
#                     grad_proj = current_gradients-(angle/length_rep)*self.reference_gradients
#                     for n, p in strategy.model.named_parameters():
#                         if p.requires_grad:
#                             n_param = p.numel()  # number of parameters in [p]
#                             p.grad.copy_(grad_proj[count:count+n_param].view_as(p))
#                             count += n_param
#             else:
#                 count = 0
#                 for (n, p) in strategy.model.named_parameters():
#                         if p.requires_grad:
#                             n_param = p.numel()  # number of parameters in [p]
#                             p.grad.copy_( p.grad + (self.reference_gradients[count:count+n_param]).view_as(p))
#                             count += n_param

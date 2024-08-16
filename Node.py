class Node:
  def __call__(self, x, *args):
    return self.forward(x, *args)

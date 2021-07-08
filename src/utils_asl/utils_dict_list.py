import collections

__all__ = ["flatten_list", "flatten_dict"]


def flatten_list(d, parent_key="", sep="_"):
  items = []
  for num, element in enumerate(d):
    new_key = parent_key + sep + str(num) if parent_key else str(num)

    if isinstance(element, collections.MutableMapping):
      items.extend(flatten_dict(element, new_key, sep=sep).items())
    else:
      if isinstance(element, list):
        if isinstance(element[0], dict):
          items.extend(flatten_list(element, new_key, sep=sep))
          continue
      items.append((new_key, element))
  return items


def flatten_dict(d, parent_key="", sep="_"):
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      if isinstance(v, list):
        if isinstance(v[0], dict):
          items.extend(flatten_list(v, new_key, sep=sep))
          continue
      items.append((new_key, v))
  return dict(items)

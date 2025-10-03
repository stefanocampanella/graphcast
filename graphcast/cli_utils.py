import click


class DictParamType(click.ParamType):
  """Click ParamType that parses mappings like "a:1,b:2" into dict[str, int].

  Rules:
  - Comma-separated items, each as key:value.
  - Keys are non-empty strings; surrounding whitespace is ignored.
  - Values must be integers; surrounding whitespace is ignored.
  - Empty string yields an empty dict.
  - Duplicate keys: later values overwrite earlier ones.

  Example:
    --param=a:1,b:2,c:3  -> {"a": 1, "b": 2, "c": 3}
  """

  name = "dict"

  def convert(self, value, param, ctx):  # type: ignore[override]
    if isinstance(value, dict):
      # Assume it's already a mapping of str->int; perform minimal validation
      result = {}
      for k, v in value.items():
        if not isinstance(k, str) or k.strip() == "":
          self.fail(f"Invalid key in mapping: {k!r}", param, ctx)
        try:
          result[k.strip()] = int(v)
        except Exception:
          self.fail(f"Invalid integer value for key {k!r}: {v!r}", param, ctx)
      return result

    if not isinstance(value, str):
      self.fail(f"Expected string for {self.name}, got {type(value).__name__}", param, ctx)

    text = value.strip()
    if text == "":
      return {}

    items = [p for p in (s.strip() for s in text.split(",")) if p != ""]
    result: dict[str, int] = {}
    for item in items:
      if ":" not in item:
        self.fail(f"Invalid item {item!r}. Expected 'key:value' pairs separated by commas.", param, ctx)
      key, val = item.split(":", 1)
      key = key.strip()
      val = val.strip()
      if key == "":
        self.fail("Empty key is not allowed in mapping.", param, ctx)
      try:
        result[key] = int(val)
      except Exception:
        self.fail(f"Value for key {key!r} must be an integer, got {val!r}.", param, ctx)
    return result

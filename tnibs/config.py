from dataclasses import dataclass, asdict
from typing import Dict


# use the decorator above
class Config:
    # autodataclass. Note that static code analyzewrs may complain in which case you may need to write the @dataclass(kw_only=True) as well
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(kw_only=True)(cls)

    @classmethod
    def _filter_kwargs(cls, *dicts, **kwargs):
        def arg_to_dict(d):
            if isinstance(d, type):  # Check if d is a class
                return vars(d)
            #  Returns False for some reason
            # if isinstance(d, Config):
            #     d = d.__dict__
            if isinstance(d, Dict):
                return d
            else:
                return d.__dict__

        all_dicts = tuple(arg_to_dict(d) for d in dicts) + (kwargs,)

        merged_dict = {
            key: value
            for d in all_dicts
            for key, value in d.items()
            if key in cls.__dataclass_fields__
        }

        return merged_dict
    

    @classmethod
    def create(cls, *dicts, **kwargs):
        """Use one or more positional Dict/Config/Class's and/or kwargs to fill in the fields of and create a Config."""
        return cls(**cls._filter_kwargs(*dicts, **kwargs))

    @classmethod
    def dict_from(cls, *dicts, **kwargs):
        """create but outputs a Dict"""
        return asdict(cls.create(*dicts, **kwargs))

    # def show(self, indent=4):
    #     """
    #     Pretty prints a (possibly deeply-nested) dataclass.
    #     Each new block will be indented by `indent` spaces (default is 4).
    #     """
    #     print(stringify(self, indent))


    # extend dataclass behavior to act like dict

    def __call__(self, *dicts, **kwargs):
        """Use one or more positional Dict/Config/Class's and/or kwargs to update only existing attributes"""
        for key, value in self._filter_kwargs(*dicts, **kwargs).items():
            setattr(self, key, value)
        return self

    def _dict(self, include_none=False):
        return {k: v for k, v in asdict(self).items() if include_none or v is not None}
    
    def __iter__(self):
        """Returns the dict representation of the instance when iterated over."""
        return iter(self._dict().items())
    
    def __getitem__(self, k):
        return asdict(self)[k]

    def keys(self):
        return self._dict().keys()


    # a hack disguising the fact that we diverge from __dataclass_fields__, but its just for updating dict()
    # not sure about whether del self.__dict__[key] does anything
    def _del(self, key, warn=True):
        if key in self.__dict__:
            setattr(self, key, None)
        else:
            if warn:
                print(f"Warning: Key '{key}' not found.")



# unused: some kind of better printing maybe
# def stringify(obj, indent=4, _indents=0):
#     if isinstance(obj, str):
#         return f"'{obj}'"

#     if not is_dataclass(obj) and not isinstance(obj, (Mapping, Iterable)):
#         return str(obj)

#     this_indent = indent * _indents * " "
#     next_indent = indent * (_indents + 1) * " "
#     start, end = (
#         f"{type(obj).__name__}(",
#         ")",
#     )  # dicts, lists, and tuples will re-assign this

#     if is_dataclass(obj):
#         body = "\n".join(
#             f"{next_indent}{field.name}="
#             f"{stringify(getattr(obj, field.name), indent, _indents + 1)},"
#             for field in fields(obj)
#         )

#     elif isinstance(obj, Mapping):
#         if isinstance(obj, dict):
#             start, end = "{}"

#         body = "\n".join(
#             f"{next_indent}{stringify(key, indent, _indents + 1)}: "
#             f"{stringify(value, indent, _indents + 1)},"
#             for key, value in obj.items()
#         )

#     else:  # is Iterable
#         if isinstance(obj, list):
#             start, end = "[]"
#         elif isinstance(obj, tuple):
#             start = "("

#         body = "\n".join(
#             f"{next_indent}{stringify(item, indent, _indents + 1)}," for item in obj
#         )

#     return f"{start}\n{body}\n{this_indent}{end}"

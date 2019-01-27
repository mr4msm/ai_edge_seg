# -*- coding: utf-8 -*-


def argv2string(argv, delimiter=' '):
    """Convert arg list to a string."""
    assert len(argv) > 0
    arg_str = argv[0]
    for arg in argv[1:]:
        arg_str += delimiter + arg
    return arg_str

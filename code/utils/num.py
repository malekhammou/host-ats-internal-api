def above_zero_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x <=0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive int"%(x,))
    return x

def above_zero_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x <= 0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def outer():  # python2.x
    share = {'var': 0}
    def inner_dec():
        share['var'] -= 1;
        print("var is {0}".format(share['var']));

    def inner_inc():
        share['var'] += 1;
        print("var is {0}".format(share['var']));
    return inner_dec, inner_inc

if __name__=='__main__':
    dec, inc = outer()
    dec()
    dec()
    dec()
    dec()
    inc()
    inc()
    inc()
    inc()


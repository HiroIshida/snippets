import traceback

try:
    x = 1 / 0    # ゼロ除算
except:
    print(traceback.format_exc())    # いつものTracebackが表示される
    traceback.print_exc()                 #

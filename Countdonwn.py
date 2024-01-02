from scamp import *

s = Session()

def on_move(x, y):
    value = int(x * 40000 if x * 40000 < 31896 else 31896)
    print(f'{value:,}')

s.register_mouse_listener(on_move=on_move, relative_coordinates=True)
import tkinter as _tk
import tkinter.ttk as _ttk
from tkinter import filedialog
import os
import shutil
from PIL import Image, ImageTk

IMAGE_POS = 0
IMAGE_MAYBE = 1
IMAGE_NEG = 2

def select_input_folder():
    folder_selected = filedialog.askdirectory()
    return folder_selected

input_path = select_input_folder()

#input_path = os.path.abspath('./input')
matches_path = os.path.abspath('./matches')
maybes_path = os.path.abspath('./maybes')
dislike_path = os.path.abspath('./dislike')

image_idx = 0
image_paths = []
for img in os.listdir(input_path):
    if img.lower().endswith((".png", ".jpg")):
        image_paths.append(os.path.join(input_path, img))

print(f"Loaded {len(image_paths)} images")

os.makedirs(matches_path, exist_ok=True)
os.makedirs(maybes_path, exist_ok=True)
os.makedirs(dislike_path, exist_ok=True)

tk = _tk.Tk()

frame = _tk.Frame(tk, borderwidth=4, relief=_tk.FLAT)
frame.pack(side=_tk.TOP, anchor=_tk.N, expand=True, fill=_tk.BOTH)

def resize_image(image, base_width=800):
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    return image.resize((base_width, h_size), Image.LANCZOS)

img = Image.open(image_paths[0])
img = resize_image(img)
active_image_preview = ImageTk.PhotoImage(img)

label_image_preview = _ttk.Label(frame, image=active_image_preview)
label_image_preview.pack(side=_tk.TOP, anchor=_tk.CENTER,
                         expand=True, fill=_tk.BOTH)

label_feedback = _ttk.Label(frame, text=f"Image {image_idx+1} of {len(image_paths)}")
label_feedback.pack(side=_tk.TOP, anchor=_tk.CENTER)

frame_buttons = _ttk.Frame(frame)
frame_buttons.pack(side=_tk.BOTTOM, anchor=_tk.S, expand=True, fill=_tk.BOTH)

button_nay = _ttk.Button(frame_buttons, text="Nay (a)")
button_maybe = _ttk.Button(frame_buttons, text="Maybe (w)")
button_yay = _ttk.Button(frame_buttons, text="Yay (d)")

button_maybe.pack(side=_tk.LEFT, anchor=_tk.W, expand=True, fill=_tk.X)
button_nay.pack(side=_tk.LEFT, anchor=_tk.W, expand=True, fill=_tk.X)
button_yay.pack(side=_tk.LEFT, anchor=_tk.W, expand=True, fill=_tk.X)

def next_image(i: int):
    global active_image_preview, image_idx, label_feedback

    active_img_path = image_paths[image_idx]
    img_name = os.path.basename(active_img_path)

    if i == IMAGE_POS:
        shutil.copyfile(active_img_path, os.path.join(matches_path, img_name))
    elif i == IMAGE_MAYBE:
        shutil.copyfile(active_img_path, os.path.join(maybes_path, img_name))
    elif i == IMAGE_NEG:
        shutil.copyfile(active_img_path, os.path.join(dislike_path, img_name))

    next_idx = image_idx + 1
    if next_idx >= len(image_paths):
        next_idx = 0

    img = Image.open(image_paths[next_idx])
    img = resize_image(img)
    active_image_preview = ImageTk.PhotoImage(img)

    label_image_preview.configure(image=active_image_preview)
    label_feedback.configure(text=f"Image {next_idx+1} of {len(image_paths)}")

    image_idx = next_idx

def key(event):
    if event.char == 'd':
        next_image(IMAGE_POS)
    elif event.char == 'w':
        next_image(IMAGE_MAYBE)
    elif event.char == 'a':
        next_image(IMAGE_NEG)

button_yay.bind('<Button-1>', lambda e: next_image(IMAGE_POS))
button_maybe.bind('<Button-1>', lambda e: next_image(IMAGE_MAYBE))
button_nay.bind('<Button-1>', lambda e: next_image(IMAGE_NEG))

tk.bind('<Key>', key)

tk.pack_slaves()
tk.mainloop()

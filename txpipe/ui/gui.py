import tkinter as tk
import sys

# pip install pillow
from PIL import Image, ImageTk
import os
import platform
from .pipeline_monitor import PipelineMonitor

# set up your Tk Frame and whatnot here...


def window_focus():
    if platform.system() == "Darwin":  # How Mac OS X is identified by Python
        os.system(
            """/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' """
        )


class ConfigWindow:
    def __init__(self, master):
        self.master = master
        master.title("Pipeline Monitor Configuration")

        self.label1 = tk.Label(master, text="NERSC Username:")
        self.entry1 = tk.Entry()

        self.label2 = tk.Label(master, text="Public key:")
        self.entry2 = tk.Entry()
        self.entry2.insert(tk.END, "~/.ssh/nersc.pub")

        self.label3 = tk.Label(master, text="Cori TXPipe directory:")
        self.entry3 = tk.Entry()

        self.label4 = tk.Label(master, text="Remote yaml file:")
        self.entry4 = tk.Entry()
        self.entry4.insert(tk.END, "test/demo.yml")

        self.button = tk.Button(master, text="Start", command=self.submit)

        self.label1.grid(row=1, column=1)
        self.entry1.grid(row=1, column=2)

        self.label2.grid(row=2, column=1)
        self.entry2.grid(row=2, column=2)

        self.label3.grid(row=3, column=1)
        self.entry3.grid(row=3, column=2)

        self.label4.grid(row=4, column=1)
        self.entry4.grid(row=4, column=2)

        self.button.grid(row=5, columnspan=2)

        self.entry1.focus_set()

    def submit(self):
        ok = True
        vals = []
        for entry in [self.entry1, self.entry2, self.entry3, self.entry4]:
            val = entry.get()
            vals.append(val)
            if val == "":
                entry.config({"background": "pink"})
                ok = False
        if ok:
            self.master.destroy()
            start_monitor(*vals, True)


class PipelineWindow:
    def __init__(self, master):
        self.master = master
        master.title("Pipeline Monitor")
        master.geometry("800x600")

        self.img = None
        self.img_resized = None

        self.background = tk.Label(master, image=self.img_resized)
        self.background.pack(side="bottom", fill="both", expand="yes")
        self.background.bind("<Configure>", self._resize_image)

        # self.canvas = tk.Canvas(master, width=self.width, height=self.height)
        # self.canvas.pack()
        # create empty image
        # self.img_frame = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_resized)
        # self.canvas.bind('<Configure>', self._resize_image)

    def set_image(self, img):
        self.img = img
        width = self.background.winfo_width()
        height = self.background.winfo_height()

        ratio = min(width / img.width, height / img.height)
        w = int(img.width * ratio)
        h = int(img.height * ratio)

        img_resized = img.resize((w, h), Image.LANCZOS)
        self.img_resized = ImageTk.PhotoImage(img_resized)
        self.background.configure(image=self.img_resized)
        # self.canvas.itemconfig(self.img_frame, image=self.img_resized)

    def _resize_image(self, event):
        if self.img is not None:
            self.set_image(self.img)
        # new_width = event.width
        # new_height = event.height

        # self.image_resized = self.img_copy.resize((new_width, new_height))
        # self.background_image = ImageTk.PhotoImage(self.image)
        # self.background.configure(image =  self.background_image)


def start_monitor(username, key_filename, remote_dir, config_file, from_gui):
    if from_gui:
        print("")
        print("You can quickstart this using this command in future:")
        print(
            f"python -m txpipe.ui {username} {key_filename} {remote_dir} {config_file}"
        )

    # start a new TK instance
    root = tk.Tk()
    window = PipelineWindow(root)
    monitor = PipelineMonitor(username, key_filename, remote_dir, config_file)
    img = monitor.draw_pil()
    first = True
    while True:
        try:
            root.update_idletasks()
            root.update()
            if first:
                window.set_image(img)
                first = False
            changes = monitor.update()
            if changes:
                img = monitor.draw_pil()
                window.set_image(img)
        except KeyboardInterrupt:
            break
        except tk._tkinter.TclError:
            break


def main():
    if len(sys.argv) == 5:
        start_monitor(*sys.argv[1:], False)
    else:
        root = tk.Tk()
        window = ConfigWindow(root)
        window_focus()
        root.mainloop()


if __name__ == "__main__":
    main()

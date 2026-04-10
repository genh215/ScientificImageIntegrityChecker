import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab


def apply_filter(img: np.ndarray, periods: int) -> np.ndarray:
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_normalized = img.astype(np.float32) / 255.0
    img_filtered = np.sin(img_normalized * 2.0 * np.pi * float(periods)) * 0.5 + 0.5
    return np.clip(img_filtered * 255.0, 0, 255).astype(np.uint8)


class ImageCheckerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Data Checker (Grayscale Comparison)")
        self.geometry("1000x700")

        self.original_image = None
        self.filtered_image = None
        self.preview_photo = None

        self._build_ui()
        self.bind("<Configure>", self._on_resize)
        self.bind("<Control-v>", self._on_ctrl_v)

    def _build_ui(self):
        self.img_frame = ttk.Frame(self)
        self.img_frame.pack(padx=10, pady=(10, 5), expand=True, fill=tk.BOTH)

        self.image_label = ttk.Label(self.img_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)

        control_frame = ttk.Frame(self)
        control_frame.pack(padx=10, pady=5, fill=tk.X)

        self.period_var = tk.IntVar(value=4)

        self.scale = tk.Scale(
            control_frame,
            from_=1,
            to=32,
            orient=tk.HORIZONTAL,
            variable=self.period_var,
            command=self.update_filter
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(btn_frame, text="Load", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load from Clipboard", command=self.load_from_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=5)

        self.status = tk.StringVar(value="No image")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def _on_resize(self, event):
        if self.original_image is not None:
            self.update_preview()

    def _on_ctrl_v(self, event):
        self.load_from_clipboard()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files",
                 "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp;*.jp2"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Error", "Cannot read image")
            return

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.original_image = img
        self.status.set(f"Loaded: {os.path.basename(path)}")

        self.update_filter()

    def load_from_clipboard(self):
        clip = ImageGrab.grabclipboard()

        if clip is None:
            messagebox.showwarning("Clipboard", "No image in clipboard")
            return

        img = np.array(clip)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.original_image = img
        self.status.set("Loaded from clipboard")

        self.update_filter()

    def save_image(self):
        if self.filtered_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("TIFF", "*.tif;*.tiff"),
                ("BMP", "*.bmp"),
                ("WEBP", "*.webp"),
                ("JPEG2000", "*.jp2"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        success = cv2.imwrite(path, self.filtered_image)

        if not success:
            messagebox.showerror("Error", "Failed to save image")

    def update_filter(self, event=None):
        if self.original_image is None:
            return

        p = self.period_var.get()
        self.filtered_image = apply_filter(self.original_image, p)
        self.update_preview()

    def resize_to_fit(self, img):
        fw = self.img_frame.winfo_width()
        fh = self.img_frame.winfo_height()

        h, w = img.shape[:2]
        scale = min(fw / w, fh / h, 1.0)

        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def to_grayscale(self, img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def make_side_by_side(self):
        h = min(self.original_image.shape[0], self.filtered_image.shape[0])
        w = min(self.original_image.shape[1], self.filtered_image.shape[1])

        orig = self.original_image[:h, :w]
        filt = self.filtered_image[:h, :w]

        orig = self.to_grayscale(orig)
        filt = self.to_grayscale(filt)

        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        filt = cv2.cvtColor(filt, cv2.COLOR_GRAY2BGR)

        cv2.putText(orig, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(filt, "Filtered", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return np.hstack([orig, filt])

    def update_preview(self):
        if self.filtered_image is None:
            return

        combined = self.make_side_by_side()
        resized = self.resize_to_fit(combined)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        self.preview_photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.preview_photo)


if __name__ == "__main__":
    app = ImageCheckerApp()
    app.mainloop()

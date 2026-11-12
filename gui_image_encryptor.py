import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from arnold_cat_map import (
    arnold_cat_map,
    logistic_map_scramble,
    logistic_map_descramble,
    inverse_arnold_cat_map,
    save_image_from_matrix
)
from image_to_rgb_matrix import image_to_rgb_matrix, crop_and_resize_to_square

class ModernImageEncryptorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AetherCipher | Classical Chaos Image Encryption")
        self.root.geometry("1100x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#0b0f19")

        # Runtime Data
        self.original_image_path = None
        self.original_matrix = None
        self.encrypted_matrix = None
        self.decrypted_matrix = None

        self.encrypted_image_path = "encrypted_image_gui.png"
        self.decrypted_image_path = "decrypted_image_gui.png"

        # Apply Modern Styles
        self.setup_styles()

        # Build UI Layout
        self.create_header()
        self.create_main_content()
        self.create_status_bar()

        # Load sample image if present
        if os.path.exists("decrypted_image_gui.png"):
            self.load_image_from_path("decrypted_image_gui.png")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Configure dark colors
        style.configure(".", background="#0b0f19", foreground="#f8fafc", font=("Helvetica", 10))
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#111827", relief="flat")
        
        style.configure("Header.TLabel", background="#0b0f19", foreground="#06b6d4", font=("Helvetica", 16, "bold"))
        style.configure("SubHeader.TLabel", background="#0b0f19", foreground="#94a3b8", font=("Helvetica", 9))
        style.configure("CardTitle.TLabel", background="#111827", foreground="#e2e8f0", font=("Helvetica", 11, "bold"))
        style.configure("CardText.TLabel", background="#111827", foreground="#94a3b8", font=("Helvetica", 9))
        style.configure("Metric.TLabel", background="#111827", foreground="#38bdf8", font=("Courier", 9, "bold"))

        # Modern Buttons
        style.configure("Primary.TButton", background="#0284c7", foreground="#ffffff", font=("Helvetica", 10, "bold"), borderwidth=0, padding=6)
        style.map("Primary.TButton", background=[("active", "#0369a1"), ("disabled", "#334155")])

        style.configure("Success.TButton", background="#059669", foreground="#ffffff", font=("Helvetica", 10, "bold"), borderwidth=0, padding=6)
        style.map("Success.TButton", background=[("active", "#047857"), ("disabled", "#334155")])

        style.configure("Secondary.TButton", background="#1e293b", foreground="#e2e8f0", font=("Helvetica", 9), borderwidth=0, padding=5)
        style.map("Secondary.TButton", background=[("active", "#334155")])

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#0b0f19", pady=10, padx=20)
        header_frame.pack(fill="x")

        title_box = tk.Frame(header_frame, bg="#0b0f19")
        title_box.pack(side="left")

        title = tk.Label(title_box, text="AetherCipher Suite", font=("Helvetica", 16, "bold"), fg="#38bdf8", bg="#0b0f19")
        title.pack(anchor="w")

        subtitle = tk.Label(title_box, text="Classical 2-Stage Chaotic Image Encryption: Arnold 2D Cat Map + Logistic Map", font=("Helvetica", 9), fg="#94a3b8", bg="#0b0f19")
        subtitle.pack(anchor="w")

        # Top Action Button
        top_btn_box = tk.Frame(header_frame, bg="#0b0f19")
        top_btn_box.pack(side="right")

        ttk.Button(top_btn_box, text="Open Image...", style="Primary.TButton", command=self.select_image).pack(side="left", padx=5)

    def create_main_content(self):
        main_frame = tk.Frame(self.root, bg="#0b0f19", padx=16, pady=6)
        main_frame.pack(fill="both", expand=True)

        # 1. Parameter Control Bar
        control_card = tk.Frame(main_frame, bg="#111827", padx=16, pady=12, highlightbackground="#1f2937", highlightthickness=1)
        control_card.pack(fill="x", pady=(0, 14))

        ctrl_inner = tk.Frame(control_card, bg="#111827")
        ctrl_inner.pack(fill="x")

        # Arnold Cat Map Rounds
        tk.Label(ctrl_inner, text="Arnold Rounds (k):", fg="#cbd5e1", bg="#111827", font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.entry_k = tk.Spinbox(ctrl_inner, from_=1, to=10, width=5, bg="#1e293b", fg="#ffffff", insertbackground="white", relief="flat", font=("Helvetica", 9))
        self.entry_k.delete(0, "end")
        self.entry_k.insert(0, "1")
        self.entry_k.grid(row=0, column=1, padx=(0, 16))

        # Logistic Map Seed x
        tk.Label(ctrl_inner, text="Logistic Seed x₀ (0 < x < 1):", fg="#cbd5e1", bg="#111827", font=("Helvetica", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.entry_x = tk.Entry(ctrl_inner, width=10, bg="#1e293b", fg="#ffffff", insertbackground="white", relief="flat", font=("Helvetica", 9))
        self.entry_x.insert(0, "0.5888")
        self.entry_x.grid(row=0, column=3, padx=(0, 16))

        # Logistic Parameter r
        tk.Label(ctrl_inner, text="Chaos Param r (3.57 ≤ r ≤ 4.0):", fg="#cbd5e1", bg="#111827", font=("Helvetica", 9, "bold")).grid(row=0, column=4, sticky="w", padx=(0, 4))
        self.entry_r = tk.Entry(ctrl_inner, width=10, bg="#1e293b", fg="#ffffff", insertbackground="white", relief="flat", font=("Helvetica", 9))
        self.entry_r.insert(0, "3.98888")
        self.entry_r.grid(row=0, column=5, padx=(0, 16))

        # Preset & Random buttons
        btn_box = tk.Frame(ctrl_inner, bg="#111827")
        btn_box.grid(row=0, column=6, sticky="e")

        ttk.Button(btn_box, text="Random Keys", style="Secondary.TButton", command=self.generate_random_keys).pack(side="left", padx=3)
        ttk.Button(btn_box, text="Standard", style="Secondary.TButton", command=lambda: self.set_keys("1", "0.5888", "3.98888")).pack(side="left", padx=3)
        ttk.Button(btn_box, text="Hyper (3x)", style="Secondary.TButton", command=lambda: self.set_keys("3", "0.2468", "3.9999")).pack(side="left", padx=3)

        # 2. Main 3-Panel Preview Area
        preview_container = tk.Frame(main_frame, bg="#0b0f19")
        preview_container.pack(fill="both", expand=True)
        preview_container.grid_columnconfigure((0, 1, 2), weight=1, uniform="group1")
        preview_container.grid_rowconfigure(0, weight=1)

        # Panel 1: Original
        self.card_orig, self.lbl_img_orig, self.lbl_meta_orig = self.create_image_card(
            preview_container, column=0, title="1. Plaintext (Original)",
            btn_text="Select Different...", btn_cmd=self.select_image
        )

        # Panel 2: Encrypted
        self.card_enc, self.lbl_img_enc, self.lbl_meta_enc = self.create_image_card(
            preview_container, column=1, title="2. Ciphertext (Encrypted)",
            btn_text="Encrypt Image", btn_cmd=self.encrypt_image,
            action_style="Primary.TButton", save_cmd=self.save_encrypted_as
        )

        # Panel 3: Decrypted
        self.card_dec, self.lbl_img_dec, self.lbl_meta_dec = self.create_image_card(
            preview_container, column=2, title="3. Plaintext (Decrypted)",
            btn_text="Decrypt Image", btn_cmd=self.decrypt_image,
            action_style="Success.TButton", save_cmd=self.save_decrypted_as
        )

    def create_image_card(self, parent, column, title, btn_text, btn_cmd, action_style="Secondary.TButton", save_cmd=None):
        card = tk.Frame(parent, bg="#111827", padx=14, pady=14, highlightbackground="#1f2937", highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=6)

        header = tk.Label(card, text=title, font=("Helvetica", 10, "bold"), fg="#f8fafc", bg="#111827")
        header.pack(anchor="w", pady=(0, 8))

        # Image view box
        img_box = tk.Frame(card, bg="#030712", highlightbackground="#1f2937", highlightthickness=1)
        img_box.pack(fill="both", expand=True)

        lbl_img = tk.Label(img_box, text="[No Image]", bg="#030712", fg="#475569", font=("Helvetica", 10))
        lbl_img.pack(fill="both", expand=True)

        # Metadata
        lbl_meta = tk.Label(card, text="Dimensions: --\nEntropy: -- | Corr: --", justify="left", fg="#94a3b8", bg="#111827", font=("Helvetica", 8))
        lbl_meta.pack(anchor="w", pady=(8, 8))

        # Button row
        btn_bar = tk.Frame(card, bg="#111827")
        btn_bar.pack(fill="x")

        ttk.Button(btn_bar, text=btn_text, style=action_style, command=btn_cmd).pack(side="left", fill="x", expand=True)
        if save_cmd:
            ttk.Button(btn_bar, text="Save As...", style="Secondary.TButton", command=save_cmd).pack(side="left", padx=(4, 0))

        return card, lbl_img, lbl_meta

    def create_status_bar(self):
        self.status_bar = tk.Frame(self.root, bg="#030712", padx=14, pady=6)
        self.status_bar.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(self.status_bar, text="Ready. Open an image or use the loaded sample to begin.", fg="#94a3b8", bg="#030712", font=("Helvetica", 9))
        self.status_lbl.pack(side="left")

        self.status_sub = tk.Label(self.status_bar, text="2-Stage Permutation Cipher", fg="#06b6d4", bg="#030712", font=("Helvetica", 9))
        self.status_sub.pack(side="right")

    def set_keys(self, k, x, r):
        self.entry_k.delete(0, "end")
        self.entry_k.insert(0, k)
        self.entry_x.delete(0, "end")
        self.entry_x.insert(0, x)
        self.entry_r.delete(0, "end")
        self.entry_r.insert(0, r)
        self.status_lbl.config(text=f"Applied preset keys: k={k}, x₀={x}, r={r}", fg="#38bdf8")

    def generate_random_keys(self):
        k = str(np.random.randint(1, 4))
        x = f"{0.05 + np.random.random() * 0.9:.4f}"
        r = f"{3.85 + np.random.random() * 0.149:.5f}"
        self.set_keys(k, x, r)
        self.status_lbl.config(text=f"Generated random keys: k={k}, x₀={x}, r={r}", fg="#c084fc")

    def select_image(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Image to Encrypt", filetypes=filetypes)
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):
        try:
            # Auto center-crop to square if needed
            self.original_matrix = crop_and_resize_to_square(path, target_size=512)
            self.original_image_path = path

            # Display original
            self.display_matrix(self.original_matrix, self.lbl_img_orig)

            # Metrics
            entropy, h_corr = self.compute_metrics(self.original_matrix)
            h, w, _ = self.original_matrix.shape
            self.lbl_meta_orig.config(text=f"Dimensions: {w}x{h} px\nEntropy: {entropy:.4f} bits | Corr: {h_corr:.4f}")

            self.status_lbl.config(text=f"Loaded image: {os.path.basename(path)} ({w}x{h})", fg="#34d399")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def encrypt_image(self):
        if self.original_matrix is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            k = int(self.entry_k.get())
            x = float(self.entry_x.get())
            r = float(self.entry_r.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric parameters for k, x, and r.")
            return

        if not (0.0 < x < 1.0):
            messagebox.showerror("Error", "Logistic seed x₀ must be strictly between 0 and 1.")
            return
        if not (3.57 <= r <= 4.0):
            messagebox.showerror("Error", "Chaotic parameter r must be between 3.57 and 4.0.")
            return

        self.status_lbl.config(text="Encrypting with Arnold Cat Map + Logistic Scrambler...", fg="#38bdf8")
        self.root.update()

        try:
            # 1. Arnold Cat Map
            scrambled = arnold_cat_map(self.original_matrix, iterations=k)
            # 2. Logistic Map
            self.encrypted_matrix = logistic_map_scramble(scrambled, x, r)

            save_image_from_matrix(self.encrypted_matrix, self.encrypted_image_path)
            self.display_matrix(self.encrypted_matrix, self.lbl_img_enc)

            entropy, h_corr = self.compute_metrics(self.encrypted_matrix)
            h, w, _ = self.encrypted_matrix.shape
            self.lbl_meta_enc.config(text=f"Dimensions: {w}x{h} px\nEntropy: {entropy:.4f} bits | Corr: {h_corr:.4f}")

            self.status_lbl.config(text=f"Encrypted successfully! (Spatial Corr: {h_corr:.4f})", fg="#38bdf8")
        except Exception as e:
            messagebox.showerror("Error", f"Encryption failed: {e}")

    def decrypt_image(self):
        matrix_to_decrypt = self.encrypted_matrix if self.encrypted_matrix is not None else self.original_matrix
        if matrix_to_decrypt is None:
            if os.path.exists(self.encrypted_image_path):
                matrix_to_decrypt = image_to_rgb_matrix(self.encrypted_image_path)
            else:
                messagebox.showerror("Error", "Please select or encrypt an image first.")
                return

        try:
            k = int(self.entry_k.get())
            x = float(self.entry_x.get())
            r = float(self.entry_r.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric parameters.")
            return

        self.status_lbl.config(text="Decrypting and inverting chaotic orbits...", fg="#34d399")
        self.root.update()

        try:
            # 1. Inverse Logistic
            descrambled = logistic_map_descramble(matrix_to_decrypt, x, r)
            # 2. Inverse Arnold
            self.decrypted_matrix = inverse_arnold_cat_map(descrambled, iterations=k)

            save_image_from_matrix(self.decrypted_matrix, self.decrypted_image_path)
            self.display_matrix(self.decrypted_matrix, self.lbl_img_dec)

            entropy, h_corr = self.compute_metrics(self.decrypted_matrix)
            h, w, _ = self.decrypted_matrix.shape

            # Verify integrity
            is_match = False
            if self.original_matrix is not None:
                is_match = np.array_equal(self.original_matrix, self.decrypted_matrix)

            match_text = "100% Bitwise Exact" if is_match else "Restored"
            self.lbl_meta_dec.config(text=f"Integrity: {match_text}\nEntropy: {entropy:.4f} | Corr: {h_corr:.4f}")

            self.status_lbl.config(text=f"Decryption complete! Match: {match_text}", fg="#34d399")
        except Exception as e:
            messagebox.showerror("Error", f"Decryption failed: {e}")

    def display_matrix(self, matrix, label_widget):
        img = Image.fromarray(np.uint8(matrix))
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def compute_metrics(self, matrix):
        try:
            flat = matrix.flatten()
            counts = np.bincount(flat, minlength=256)
            probs = counts[counts > 0] / len(flat)
            entropy = float(-np.sum(probs * np.log2(probs)))

            gray = 0.2989 * matrix[:, :, 0] + 0.5870 * matrix[:, :, 1] + 0.1140 * matrix[:, :, 2]
            x_h = gray[:, :-1].flatten()
            y_h = gray[:, 1:].flatten()

            samples = 3000
            if len(x_h) > samples:
                idx = np.random.choice(len(x_h), samples, replace=False)
                x_h, y_h = x_h[idx], y_h[idx]

            cov = np.cov(x_h, y_h)[0, 1]
            std_x, std_y = np.std(x_h), np.std(y_h)
            corr = float(cov / (std_x * std_y)) if std_x * std_y != 0 else 0.0
            return entropy, corr
        except Exception:
            return 0.0, 0.0

    def save_encrypted_as(self):
        if self.encrypted_matrix is None:
            messagebox.showerror("Error", "No encrypted image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            save_image_from_matrix(self.encrypted_matrix, path)
            messagebox.showinfo("Saved", f"Saved encrypted image to:\n{path}")

    def save_decrypted_as(self):
        if self.decrypted_matrix is None:
            messagebox.showerror("Error", "No decrypted image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if path:
            save_image_from_matrix(self.decrypted_matrix, path)
            messagebox.showinfo("Saved", f"Saved decrypted image to:\n{path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernImageEncryptorGUI(root)
    root.mainloop()

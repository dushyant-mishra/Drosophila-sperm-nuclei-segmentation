
print("Initializing GUI...")
import tkinter as tk
print("Imported tkinter")
from tkinter import filedialog, messagebox, ttk
import os
import glob
print("Importing scientific libraries...")
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Import the segmentation logic
print("Importing segmentation logic...")
import sperm_segmentation as segmentation
print("Imports complete.")

class SpermGUI:
    def __init__(self, root):
        print("Creating main window...")
        self.root = root
        self.root.title("Sperm Segmentation Tool")
        self.root.geometry("1400x900")

        # Program State
        self.input_dir = ""
        self.files = []
        self.current_idx = 0
        self.current_img = None
        self.current_overlay = None

        # ROI State
        self.roi_points = []
        self.drawing = False
        self.roi_mask = None
        self.roi_active = False

        # --- GUI Layout ---
        
        # Sidebar
        self.sidebar = tk.Frame(root, width=250, bg="#f0f0f0")
        self.sidebar.pack(side="left", fill="y")
        
        # Load Button
        tk.Button(self.sidebar, text="Load Directory", command=self.load_directory, height=2).pack(fill="x", padx=5, pady=5)
        
        # Status Label
        self.lbl_status = tk.Label(self.sidebar, text="No directory loaded", wraplength=240)
        self.lbl_status.pack(pady=5)

        # Z-Stack Navigation
        tk.Label(self.sidebar, text="Z-Slice Navigation").pack(pady=(20, 0))
        self.scale_z = tk.Scale(self.sidebar, from_=0, to=0, orient="horizontal", command=self.on_slide_change)
        self.scale_z.pack(fill="x", padx=10)
        self.lbl_z = tk.Label(self.sidebar, text="Z: 0 / 0")
        self.lbl_z.pack()
        
        # Interaction Modes
        tk.Label(self.sidebar, text="Tools").pack(pady=(20, 5))
        self.mode_var = tk.StringVar(value="view")
        tk.Radiobutton(self.sidebar, text="View/Nav", variable=self.mode_var, value="view").pack(anchor="w", padx=10)
        tk.Radiobutton(self.sidebar, text="Draw ROI (Freehand)", variable=self.mode_var, value="roi").pack(anchor="w", padx=10)
        
        # Actions
        tk.Button(self.sidebar, text="Run Analysis on Slice", command=self.run_analysis_slice).pack(fill="x", padx=5, pady=20)
        
        tk.Button(self.sidebar, text="Reset ROI", command=self.reset_roi).pack(fill="x", padx=5, pady=5)

        # Main Canvas Area
        self.canvas_frame = tk.Frame(root, bg="black")
        self.canvas_frame.pack(side="right", expand=True, fill="both")
        
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bind events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def load_directory(self):
        d = filedialog.askdirectory(initialdir=segmentation.DEFAULTS['input_dir'])
        if not d: return
        self.input_dir = d
        self.files = sorted(glob.glob(os.path.join(d, "*.tif"))) # Simple match first
        
        if not self.files:
            # Try default pattern
            try:
                self.files, _ = segmentation.load_slices(d, segmentation.DEFAULTS['file_pattern'])
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return

        self.scale_z.config(to=len(self.files)-1)
        self.current_idx = 0
        self.load_image()
        self.lbl_status.config(text=f"Loaded: {os.path.basename(d)}\n{len(self.files)} slices")

    def load_image(self):
        if not self.files: return
        fpath = self.files[self.current_idx]
        self.current_img = tifffile.imread(fpath)
        
        self.render()
        self.lbl_z.config(text=f"Z: {self.current_idx} / {len(self.files)-1}")

    def on_slide_change(self, val):
        self.current_idx = int(val)
        self.load_image()

    def render(self):
        self.ax.clear()
        self.ax.axis('off')
        
        if self.current_img is not None:
            # Contrast stretch for display
            img = self.current_img.astype(float)
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99.5)
            img = np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)
            
            self.ax.imshow(img, cmap='gray')
            
            # Draw ROI if exists
            if self.roi_points:
                pts = np.array(self.roi_points)
                self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2)
                
            self.canvas.draw()

    # --- Interaction ---
    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.mode_var.get() == "roi":
            self.drawing = True
            self.roi_points = [[event.xdata, event.ydata]]

    def on_drag(self, event):
        if not self.drawing or event.inaxes != self.ax: return
        self.roi_points.append([event.xdata, event.ydata])
        
        # Fast redraw to show line
        pts = np.array(self.roi_points)
        self.ax.plot(pts[-2:, 0], pts[-2:, 1], 'r-')
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.drawing:
            self.drawing = False
            # Close the loop
            if len(self.roi_points) > 2:
                self.roi_points.append(self.roi_points[0])
                self.roi_active = True
            self.render()

    def reset_roi(self):
        self.roi_points = []
        self.roi_active = False
        self.render()

    def run_analysis_slice(self):
        if self.current_img is None: return
        
        # Prepare params
        params = segmentation.DEFAULTS.copy()
        
        try:
            # Create mask from ROI if active
            mask = None
            if self.roi_active and len(self.roi_points) > 3:
                from matplotlib.path import Path
                h, w = self.current_img.shape
                y, x = np.mgrid[:h, :w]
                points = np.transpose((x.ravel(), y.ravel()))
                path = Path(self.roi_points)
                mask = path.contains_points(points).reshape(h, w)
            
            # --- Run analysis on masked image ---
            seg = segmentation.segment_slice(self.current_img, params, roi_mask=mask)
            
            # Visualize results
            # create overlay
            overlay = segmentation.make_overlay(self.current_img, seg, params['um_per_px_xy'])
            
            # Show in new window
            top = tk.Toplevel(self.root)
            top.title(f"Results Z={self.current_idx}")
            
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.imshow(overlay)
            ax.axis('off')
            
            can = FigureCanvasTkAgg(fig, master=top)
            can.get_tk_widget().pack()
            
            lbl = tk.Label(top, text=f"Found {len(seg['results'])} spermatids")
            lbl.pack()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

if __name__ == "__main__":
    print("Starting Tkinter root...")
    root = tk.Tk()
    app = SpermGUI(root)
    print("Entering mainloop...")
    root.mainloop()


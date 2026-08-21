"""Compact graphical launcher for Saturn v5.7.1 parameter tuning workflows."""

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
TUNER = ROOT / "utils" / "tune_parameters_Saturnv5_7_1.py"
MIXED_RUNNER = ROOT / "scripts" / "run_v57_mixed_unet_primary_tuning.py"
DEFAULT_OUTPUT = ROOT / "parameter_tuning_results_v5_7" / "gui_runs"

MODE_LABELS = {
    "Preprocessing profile": "profile",
    "Classical 2D segmentation": "segmentation",
    "U-Net rescue (legacy hybrid)": "unet_rescue",
    "U-Net-primary 2D segmentation": "unet_primary",
    "Classical cross-slice tracking": "tracking",
    "U-Net-primary global tracking": "unet_primary_tracking",
}
UNET_MODES = {"unet_rescue", "unet_primary", "unet_primary_tracking"}


def build_single_stack_command(values, python_executable=None):
    """Build a validated command for the core v5.7.1 tuner."""
    python_executable = python_executable or sys.executable
    mode = MODE_LABELS.get(values["mode_label"], values["mode_label"])
    image_dir = Path(values["image_dir"]).expanduser()
    roi_path = Path(values["roi_path"]).expanduser()
    output_dir = Path(values["output_dir"]).expanduser()
    if not image_dir.is_dir():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not roi_path.is_file():
        raise ValueError(f"ROI mask not found: {roi_path}")
    if mode in UNET_MODES:
        checkpoint = Path(values["checkpoint"]).expanduser()
        if not checkpoint.is_file():
            raise ValueError(f"U-Net checkpoint not found: {checkpoint}")
    command = [
        python_executable,
        "-u",
        str(TUNER),
        "--mode",
        mode,
        "--dir",
        str(image_dir.resolve()),
        "--slices",
        str(values.get("slices", "auto")).strip() or "auto",
        "--roi-mask",
        str(roi_path.resolve()),
        "--outdir",
        str(output_dir.resolve()),
        "--maxiter",
        str(int(values.get("candidates", 12))),
        "--review-candidates",
        str(int(values.get("review_candidates", 8))),
        "--seed",
        str(int(values.get("seed", 12345))),
    ]
    base_profile = str(values.get("base_profile", "")).strip()
    if base_profile:
        profile_path = Path(base_profile).expanduser()
        if not profile_path.is_file():
            raise ValueError(f"Base analysis profile not found: {profile_path}")
        command.extend(["--base-params", str(profile_path.resolve())])
    checkpoint_text = str(values.get("checkpoint", "")).strip()
    if checkpoint_text:
        command.extend(
            ["--unet-model", str(Path(checkpoint_text).expanduser().resolve())]
        )
    exclusion_text = str(values.get("exclusion_mask", "")).strip()
    if exclusion_text:
        exclusion_path = Path(exclusion_text).expanduser()
        if not exclusion_path.is_file():
            raise ValueError(f"Exclusion mask not found: {exclusion_path}")
        command.extend(["--exclusion-mask", str(exclusion_path.resolve())])
    metadata_text = str(values.get("metadata_xml", "")).strip()
    if metadata_text:
        metadata_path = Path(metadata_text).expanduser()
        if not metadata_path.is_file():
            raise ValueError(f"Metadata XML not found: {metadata_path}")
        command.extend(["--metadata-xml", str(metadata_path.resolve())])
    elif values.get("auto_calibration", True):
        command.append("--auto-calibration")
    if values.get("rebuild_cache", False) and mode in UNET_MODES:
        command.append("--rebuild-unet-cache")
    if mode == "unet_primary_tracking":
        command.extend(
            ["--unet-primary-tracking-backend", "global_assignment"]
        )
    return command


def build_mixed_command(values, python_executable=None):
    """Build a validated balanced multi-specimen tuning command."""
    python_executable = python_executable or sys.executable
    required = {
        "manifest": "Mixed tuning manifest",
        "checkpoint": "U-Net checkpoint",
        "base_profile": "Base analysis profile",
    }
    resolved = {}
    for key, label in required.items():
        path = Path(str(values.get(key, "")).strip()).expanduser()
        if not path.is_file():
            raise ValueError(f"{label} not found: {path}")
        resolved[key] = path.resolve()
    output_root = Path(values["output_root"]).expanduser().resolve()
    command = [
        python_executable,
        "-u",
        str(MIXED_RUNNER),
        "--manifest",
        str(resolved["manifest"]),
        "--checkpoint",
        str(resolved["checkpoint"]),
        "--base-preset",
        str(resolved["base_profile"]),
        "--output-root",
        str(output_root),
        "--segmentation-candidates",
        str(int(values.get("segmentation_candidates", 24))),
        "--tracking-candidates",
        str(int(values.get("tracking_candidates", 24))),
        "--tracking-slice-count",
        str(int(values.get("tracking_slice_count", 5))),
        "--seed",
        str(int(values.get("seed", 12345))),
    ]
    if values.get("validate_only", False):
        command.append("--validate-only")
    return command


class TunerWorkspace:
    def __init__(self, root, defaults):
        self.root = root
        self.root.title("Saturn v5.7.1 Tuning Workspace")
        self.root.geometry("980x760")
        self.process = None
        self.output_queue = queue.Queue()

        self.status_var = tk.StringVar(value="Ready")
        self.command_var = tk.StringVar(value="")
        self.progress = ttk.Progressbar(root, mode="indeterminate")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="x", padx=10, pady=(10, 4))
        self.single_tab = ttk.Frame(notebook)
        self.mixed_tab = ttk.Frame(notebook)
        notebook.add(self.single_tab, text="One Stack")
        notebook.add(self.mixed_tab, text="Balanced Multi-Sample")

        self._build_single_tab(defaults)
        self._build_mixed_tab(defaults)

        command_frame = ttk.LabelFrame(root, text="Command")
        command_frame.pack(fill="x", padx=10, pady=4)
        ttk.Entry(
            command_frame,
            textvariable=self.command_var,
            state="readonly",
        ).pack(fill="x", padx=6, pady=6)

        action_row = ttk.Frame(root)
        action_row.pack(fill="x", padx=10, pady=4)
        self.run_button = ttk.Button(
            action_row,
            text="Run Selected Tuning",
            command=lambda: self._start(notebook.index(notebook.select())),
        )
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(
            action_row,
            text="Stop",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=6)
        self.progress.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(root, textvariable=self.status_var).pack(
            fill="x", padx=10, pady=(0, 4)
        )

        log_frame = ttk.LabelFrame(root, text="Live Output")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log = tk.Text(log_frame, wrap="word", height=18)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.root.after(100, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    @staticmethod
    def _path_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(parent, text="Browse", command=browse_command).grid(
            row=row, column=2, padx=6, pady=4
        )

    def _choose_dir(self, variable):
        path = filedialog.askdirectory(initialdir=variable.get() or str(ROOT))
        if path:
            variable.set(path)

    def _choose_file(self, variable, filetypes):
        current = variable.get()
        initial = (
            str(Path(current).parent)
            if current and Path(current).parent.is_dir()
            else str(ROOT)
        )
        path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=filetypes,
        )
        if path:
            variable.set(path)

    def _build_single_tab(self, defaults):
        frame = self.single_tab
        frame.columnconfigure(1, weight=1)
        self.single = {
            "mode_label": tk.StringVar(value="U-Net-primary 2D segmentation"),
            "image_dir": tk.StringVar(value=defaults.get("image_dir", "")),
            "roi_path": tk.StringVar(value=defaults.get("roi_path", "")),
            "exclusion_mask": tk.StringVar(value=""),
            "metadata_xml": tk.StringVar(value=""),
            "base_profile": tk.StringVar(value=defaults.get("base_profile", "")),
            "checkpoint": tk.StringVar(value=defaults.get("checkpoint", "")),
            "output_dir": tk.StringVar(
                value=str(DEFAULT_OUTPUT / "single_stack")
            ),
            "slices": tk.StringVar(value="auto"),
            "candidates": tk.IntVar(value=12),
            "review_candidates": tk.IntVar(value=8),
            "seed": tk.IntVar(value=12345),
            "auto_calibration": tk.BooleanVar(value=True),
            "rebuild_cache": tk.BooleanVar(value=False),
        }
        ttk.Label(frame, text="Operation").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.single["mode_label"],
            values=list(MODE_LABELS),
            state="readonly",
        ).grid(row=0, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        self._path_row(
            frame, 1, "Image directory", self.single["image_dir"],
            lambda: self._choose_dir(self.single["image_dir"]),
        )
        self._path_row(
            frame, 2, "ROI mask", self.single["roi_path"],
            lambda: self._choose_file(
                self.single["roi_path"],
                [("ROI masks", "*.npy *.tif *.tiff"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 3, "Exclusion mask", self.single["exclusion_mask"],
            lambda: self._choose_file(
                self.single["exclusion_mask"],
                [("Masks", "*.npy *.tif *.tiff"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 4, "Base analysis profile", self.single["base_profile"],
            lambda: self._choose_file(
                self.single["base_profile"],
                [("JSON", "*.json"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 5, "U-Net checkpoint", self.single["checkpoint"],
            lambda: self._choose_file(
                self.single["checkpoint"],
                [("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 6, "Metadata XML", self.single["metadata_xml"],
            lambda: self._choose_file(
                self.single["metadata_xml"],
                [("XML", "*.xml"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 7, "Output directory", self.single["output_dir"],
            lambda: self._choose_dir(self.single["output_dir"]),
        )
        compact = ttk.Frame(frame)
        compact.grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
        for column, (label, key, width) in enumerate(
            (
                ("Slices", "slices", 13),
                ("Candidates", "candidates", 7),
                ("Review", "review_candidates", 6),
                ("Seed", "seed", 8),
            )
        ):
            ttk.Label(compact, text=label).grid(row=0, column=column * 2)
            ttk.Entry(
                compact,
                textvariable=self.single[key],
                width=width,
            ).grid(row=0, column=column * 2 + 1, padx=(3, 12))
        ttk.Checkbutton(
            frame,
            text="Automatic microscope calibration",
            variable=self.single["auto_calibration"],
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(
            frame,
            text="Rebuild U-Net probability cache",
            variable=self.single["rebuild_cache"],
        ).grid(row=9, column=2, sticky="e", padx=6, pady=4)

    def _build_mixed_tab(self, defaults):
        frame = self.mixed_tab
        frame.columnconfigure(1, weight=1)
        self.mixed = {
            "manifest": tk.StringVar(value=defaults.get("manifest", "")),
            "checkpoint": tk.StringVar(value=defaults.get("checkpoint", "")),
            "base_profile": tk.StringVar(value=defaults.get("base_profile", "")),
            "output_root": tk.StringVar(value=str(DEFAULT_OUTPUT / "mixed")),
            "segmentation_candidates": tk.IntVar(value=24),
            "tracking_candidates": tk.IntVar(value=24),
            "tracking_slice_count": tk.IntVar(value=5),
            "seed": tk.IntVar(value=12345),
            "validate_only": tk.BooleanVar(value=False),
        }
        self._path_row(
            frame, 0, "Mixed tuning manifest", self.mixed["manifest"],
            lambda: self._choose_file(
                self.mixed["manifest"],
                [("CSV", "*.csv"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 1, "Base analysis profile", self.mixed["base_profile"],
            lambda: self._choose_file(
                self.mixed["base_profile"],
                [("JSON", "*.json"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 2, "U-Net checkpoint", self.mixed["checkpoint"],
            lambda: self._choose_file(
                self.mixed["checkpoint"],
                [("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")],
            ),
        )
        self._path_row(
            frame, 3, "Output root", self.mixed["output_root"],
            lambda: self._choose_dir(self.mixed["output_root"]),
        )
        compact = ttk.Frame(frame)
        compact.grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=8)
        for column, (label, key, width) in enumerate(
            (
                ("2D candidates", "segmentation_candidates", 7),
                ("Tracking candidates", "tracking_candidates", 7),
                ("Tracking slices", "tracking_slice_count", 6),
                ("Seed", "seed", 8),
            )
        ):
            ttk.Label(compact, text=label).grid(row=0, column=column * 2)
            ttk.Entry(
                compact,
                textvariable=self.mixed[key],
                width=width,
            ).grid(row=0, column=column * 2 + 1, padx=(3, 12))
        ttk.Checkbutton(
            frame,
            text="Validate inputs only",
            variable=self.mixed["validate_only"],
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=4)

    @staticmethod
    def _values(variables):
        return {key: variable.get() for key, variable in variables.items()}

    def _start(self, selected_tab):
        if self.process is not None and self.process.poll() is None:
            return
        try:
            command = (
                build_single_stack_command(self._values(self.single))
                if selected_tab == 0
                else build_mixed_command(self._values(self.mixed))
            )
        except Exception as exc:
            messagebox.showerror("Tuning Inputs", str(exc), parent=self.root)
            return
        self.command_var.set(subprocess.list2cmdline(command))
        self.log.delete("1.0", "end")
        self.status_var.set("Running")
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress.start(12)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(("line", line))
        return_code = self.process.wait()
        self.output_queue.put(("done", return_code))

    def _drain_output(self):
        try:
            while True:
                event, value = self.output_queue.get_nowait()
                if event == "line":
                    self.log.insert("end", value)
                    self.log.see("end")
                else:
                    self.progress.stop()
                    self.run_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set(
                        "Complete" if value == 0 else f"Stopped or failed ({value})"
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._drain_output)

    def _stop(self):
        if self.process is not None and self.process.poll() is None:
            self.status_var.set("Stopping")
            self.process.terminate()

    def _close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno(
                "Tuning Is Running",
                "Stop the current tuning process and close?",
                parent=self.root,
            ):
                return
            self.process.terminate()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="")
    parser.add_argument("--roi-mask", default="")
    parser.add_argument("--base-params", default="")
    parser.add_argument("--unet-model", default="")
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    TunerWorkspace(
        root,
        {
            "image_dir": args.dir,
            "roi_path": args.roi_mask,
            "base_profile": args.base_params,
            "checkpoint": args.unet_model,
            "manifest": args.manifest,
        },
    )
    root.mainloop()


if __name__ == "__main__":
    main()

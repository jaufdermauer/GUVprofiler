import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import czifile
from aicspylibczi import CziFile
import tifffile
from PIL import Image, ImageTk
import json
import os
import time
from matplotlib.colors import PowerNorm
import faulthandler, sys
faulthandler.enable(all_threads=True)
# optional: in eine Datei loggen
faulthandler.enable(open("crash.log", "w"), all_threads=True)

np.set_printoptions(threshold=np.inf, linewidth=200)

def bbox_to_region(box):
    # Case 1: box hat x, y, w, h
    for attrs in [("x", "y", "w", "h"), ("X", "Y", "W", "H")]:
        if all(hasattr(box, a) for a in attrs):
            x, y, w, h = (int(getattr(box, a)) for a in attrs)
            return (x, y, w, h)

    # Case 2: box hat left, top, right, bottom
    for attrs in [("left", "top", "right", "bottom"), ("Left", "Top", "Right", "Bottom")]:
        if all(hasattr(box, a) for a in attrs):
            l, t, r, b = (int(getattr(box, a)) for a in attrs)
            return (l, t, r - l, b - t)

    # Case 3: box hat x0, y0, x1, y1
    for attrs in [("x0", "y0", "x1", "y1"), ("X0", "Y0", "X1", "Y1")]:
        if all(hasattr(box, a) for a in attrs):
            x0, y0, x1, y1 = (int(getattr(box, a)) for a in attrs)
            return (x0, y0, x1 - x0, y1 - y0)

    raise TypeError(f"Don't know how to convert BBox to region. Got: {box} with attrs {dir(box)}")

def pad_to(arr, tx, ty):
    x, y = arr.shape[-2], arr.shape[-1]
    pad = [(0,0)] * arr.ndim
    pad[-2] = (0, tx - x)
    pad[-1] = (0, ty - y)
    return np.pad(arr, pad, mode="constant", constant_values=0)

def to_stxy(imgs_st):
    """
    imgs_st: nested list imgs_st[s][t] -> np.ndarray with shape (1,1,x,y) (as you showed)
    returns: np.ndarray (s,t,x,y) padded to max_x/max_y across all frames in this imgs_st
    """
    max_x = max(imgs_st[s][t].shape[-2] for s in range(len(imgs_st)) for t in range(len(imgs_st[s])))
    max_y = max(imgs_st[s][t].shape[-1] for s in range(len(imgs_st)) for t in range(len(imgs_st[s])))

    out_s = []
    for s in range(len(imgs_st)):
        out_t = []
        for t in range(len(imgs_st[s])):
            a = pad_to(imgs_st[s][t], max_x, max_y)
            a2d = a[0, 0, :, :]          # (1,1,x,y) -> (x,y)
            out_t.append(a2d)
        out_s.append(np.stack(out_t, axis=0))  # (t,x,y)

    return np.stack(out_s, axis=0)            # (s,t,x,y)

# Class to create the GUI
class CZIAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUV profiler for CZI and TIF files")
        self.files = []
        self.dye_intensity_results = {}
        self.image_results = {}
        self._last_width = None
        self._last_height = None
        self.root.minsize(800, 600)
        self.root.bind("<Configure>", self.on_resize)
        self.GUVchannel = 0
        self.dye_channel = 0

        self.create_widgets()
        
    def extract_time(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None  # falls keine Zahl gefunden wird
    
    # Resizes the image when the window size changes, with debouncing.
    def on_resize(self, event):
        selected = self.tree.selection()
        if selected and hasattr(self.image_label, "image") and self.image_label.image is not None:
            iid = selected[0]
            kind = iid.split("::", 1)[0]  # "file" | "sample" | "time"
            if kind == "time":
                _, file_, sample, guv_channel, dye_channel, time_point = iid.split("::")  # ["time", file, sample, guv, dye, t]
                # Prevent frequent updates using debouncing
                current_time = time.time()
                if hasattr(self, "_last_resize_time"):
                    if current_time - self._last_resize_time < 0.2:
                        return
                self._last_resize_time = current_time

                new_width = self.image_label.winfo_width()
                new_height = self.image_label.winfo_height()

                if hasattr(self, "_last_width") and hasattr(self, "_last_height"):
                    if self._last_width == new_width and self._last_height == new_height:
                        return

                self._last_width = new_width
                self._last_height = new_height

                self.show_image(file_, sample, guv_channel, dye_channel, time_point)
    ""
    def create_widgets(self):
        # Left side: all UI elements except the image
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="ns")

        middle_frame = tk.Frame(self.root)
        middle_frame.grid(row=0, column=1, sticky="n")

        # Right side: image display
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=2, sticky="nsew")

        # Left panel - TreeView and buttons
        self.tree = ttk.Treeview(left_frame)
        self.tree.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Scrollbar for TreeView
        self.tree_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=self.tree_scrollbar.set)

        # Image properties
        i_row = 0
        self.threshold_label = tk.Label(middle_frame, text="Image input:")
        self.threshold_label.grid(row=i_row, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        i_row+=1

        self.GUV_channels_label = tk.Label(middle_frame, text="Number of GUV channels:")
        self.GUV_channels_label.grid(row=i_row, column=0, padx=5, pady=5, sticky="ew")

        self.GUV_channels = tk.Entry(middle_frame)
        self.GUV_channels.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.GUV_channels.insert(0, "2")
        i_row+=1

        self.dye_channels_label = tk.Label(middle_frame, text="Number of dye channels:")
        self.dye_channels_label.grid(row=i_row, column=0, padx=5, pady=5, sticky="ew")

        self.dye_channels = tk.Entry(middle_frame)
        self.dye_channels.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.dye_channels.insert(0, "1")
        i_row+=1

        # Button to add files
        self.add_file_button = tk.Button(middle_frame, text="Add file", command=self.add_file)
        self.add_file_button.grid(row=i_row, column=0, columnspan=2, padx=7, pady=10, sticky="ew")
        i_row+=1

        # Threshold inputs for analysis
        self.threshold_label = tk.Label(middle_frame, text="Threshold input:")
        self.threshold_label.grid(row=i_row, column=0, padx=2, pady=10, sticky="ew")
        i_row+=1

        self.mean_intensity_label = tk.Label(middle_frame, text="Maximum lipid dye intensity inside GUV:")
        self.mean_intensity_label.grid(row=i_row, column=0, padx=17, pady=5, sticky="ew")

        self.mean_intensity_entry = tk.Entry(middle_frame)
        self.mean_intensity_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.mean_intensity_entry.insert(0, "100")
        i_row+=1

        self.circle_radius_lower_label = tk.Label(middle_frame, text="Circle Radius lower Threshold:")
        self.circle_radius_lower_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.circle_radius_lower_entry = tk.Entry(middle_frame)
        self.circle_radius_lower_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.circle_radius_lower_entry.insert(0, "5")
        i_row+=1

        self.circle_radius_upper_label = tk.Label(middle_frame, text="Circle Radius upper Threshold:")
        self.circle_radius_upper_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.circle_radius_upper_entry = tk.Entry(middle_frame)
        self.circle_radius_upper_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.circle_radius_upper_entry.insert(0, "50")
        i_row+=1

        self.circle_param2_label = tk.Label(middle_frame, text="Detection sensitivity:")
        self.circle_param2_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.circle_param2_entry = tk.Entry(middle_frame)
        self.circle_param2_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.circle_param2_entry.insert(0, "50")
        i_row+=1

        self.circle_radius_distance_tolerance_label = tk.Label(middle_frame, text="Circle radius tolerance:")
        self.circle_radius_distance_tolerance_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.circle_radius_distance_tolerance_entry = tk.Entry(middle_frame)
        self.circle_radius_distance_tolerance_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.circle_radius_distance_tolerance_entry.insert(0, "0.9")
        i_row+=1

        self.center_distance_tolerance_abs_label = tk.Label(middle_frame, text="Circle distance tolerance (absolute):")
        self.center_distance_tolerance_abs_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.center_distance_tolerance_abs_entry = tk.Entry(middle_frame)
        self.center_distance_tolerance_abs_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.center_distance_tolerance_abs_entry.insert(0, "15")
        i_row+=1

        self.center_distance_tolerance_rel_label = tk.Label(middle_frame, text="Circle distance tolerance / GUV size:")
        self.center_distance_tolerance_rel_label.grid(row=i_row, column=0, padx=7, pady=5, sticky="ew")

        self.center_distance_tolerance_rel_entry = tk.Entry(middle_frame)
        self.center_distance_tolerance_rel_entry.grid(row=i_row, column=1, padx=2, pady=5, sticky="ew")
        self.center_distance_tolerance_rel_entry.insert(0, "0.1")
        i_row+=1

        self.ch_guv_combo_var = tk.StringVar(value="All channels")
        self.ch_guv_combo = ttk.Combobox(middle_frame, textvariable=self.ch_guv_combo_var, state="readonly")
        self.ch_guv_combo.bind("<<ComboboxSelected>>", self.on_change_guv_channel)
        self.ch_guv_combo.grid(row=i_row, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        i_row+=1

        self.ch_dye_combo_var = tk.StringVar(value="All channels")
        self.ch_dye_combo = ttk.Combobox(middle_frame, textvariable=self.ch_dye_combo_var, state="readonly")
        self.ch_dye_combo.bind("<<ComboboxSelected>>", self.on_change_dye_channel)
        self.ch_dye_combo.grid(row=i_row, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        i_row+=1

        self.GUV_channels.bind("<FocusOut>", self.update_channel_comboboxes)
        self.dye_channels.bind("<FocusOut>", self.update_channel_comboboxes)
        self.update_channel_comboboxes()

        # Button for analysis
        self.analyze_button = tk.Button(middle_frame, text="Analyze", command=self.analyze_files)
        self.analyze_button.grid(row=i_row, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
        i_row+=1

        # Right side - image display
        self.image_label = tk.Label(right_frame, bg="grey")  # Hintergrundfarbe zum Debugging hinzugefügt
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Placeholder image (white) to display initially
        placeholder_img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        placeholder_photo = ImageTk.PhotoImage(placeholder_img)
        self.image_label.config(image=placeholder_photo)
        self.image_label.image = placeholder_photo  # Prevent garbage collection

        # Haupt-Container (root)
        self.root.grid_columnconfigure(0, weight=1)  # links
        self.root.grid_columnconfigure(1, weight=1)  # mitte
        self.root.grid_columnconfigure(2, weight=3)  # rechts – Bild
        self.root.grid_rowconfigure(0, weight=1)     # das ist entscheidend!

        # Configure the right panel to grow
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Configure the left panel for flexible layout
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)

        # Configure the left panel for flexible layout
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

    #Event handler when an item is selected in the TreeView
    def on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        iid = selected[0]

        kind = iid.split("::", 1)[0]  # "file" | "sample" | "channel" | "time"

        if kind == "time":
            # time -> sample -> file
            print("timepoint selected")
            channel_iid = self.tree.parent(iid)
            sample_iid = self.tree.parent(channel_iid)
            file_iid = self.tree.parent(sample_iid)

            _, file_, sample, guv_channel, dye_channel, t = iid.split("::")  # ["time", file, sample, guv, dye, t]
            t = int(t)
            intensity = self.dye_intensity_results[file_][int(sample)][int(guv_channel)][int(dye_channel)]["average_intensity"][t]
            print("file_", file_)
            self.show_image(file_, sample, guv_channel, dye_channel, t)

        elif kind == "guvchannel":
            _, file_, sample, guv_channel = iid.split("::")
            print(f"GUV channel {guv_channel} in sample {sample} of file {file_} selected")

        elif kind == "dyechannel":
            _, file_, sample, guv_channel, dye_channel = iid.split("::")
            print(f"Dye channel {dye_channel} in GUV channel {guv_channel} of sample {sample} in file {file_} selected")

        elif kind == "sample":
            # sample clicked
            _, file_, sample = iid.split("::")
            sample = int(sample)
            print(f"Sample {sample} in file {file_} selected")

        elif kind == "file":
            _, file_ = iid.split("::")
            print(f"File {file_} selected")

    # Displays the saved image for the selected file and time
    def show_image(self, file, sample, GUVchannel, dye_channel, time):
        selected_time_point = int(time)
        sample = int(sample)
        GUVchannel = int(GUVchannel)
        dye_channel = int(dye_channel)
        if file in self.image_results and sample in self.image_results[file] and GUVchannel in self.image_results[file][sample] and dye_channel in self.image_results[file][sample][GUVchannel] and selected_time_point in self.image_results[file][sample][GUVchannel][dye_channel]:
            image = self.image_results[file][sample][GUVchannel][dye_channel][selected_time_point]
            img = Image.fromarray(image)

            # Wait until the label is visible to get the correct size
            self.image_label.update_idletasks()
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            # Current image size
            img_width, img_height = img.size

            # Calculate the aspect ratio
            img_aspect_ratio = img_width / img_height
            label_aspect_ratio = label_width / label_height

            # Compare aspect ratios and scale the image
            if label_aspect_ratio > img_aspect_ratio:
                new_height = label_height
                new_width = int(label_height * img_aspect_ratio)
            else:
                new_width = label_width
                new_height = int(label_width / img_aspect_ratio)

            # Scale image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img  # Prevent garbage collection

    def _get_channel_count(self, entry_widget):
        try:
            return max(1, int(entry_widget.get()))
        except ValueError:
            return 1

    def _build_channel_options(self, count):
        return ["All channels"] + [f"Channel {i}" for i in range(1, count + 1)]

    def _selection_to_index(self, selection):
        if selection == "All channels":
            return 0
        if selection.startswith("Channel "):
            try:
                return int(selection.split(" ")[-1])
            except ValueError:
                return 0
        return 0

    def update_channel_comboboxes(self, event=None):
        guv_options = self._build_channel_options(self._get_channel_count(self.GUV_channels))
        dye_options = self._build_channel_options(self._get_channel_count(self.dye_channels))

        self.ch_guv_combo.configure(values=guv_options)
        self.ch_dye_combo.configure(values=dye_options)

        if self.ch_guv_combo_var.get() not in guv_options:
            self.ch_guv_combo_var.set("All channels")
        if self.ch_dye_combo_var.get() not in dye_options:
            self.ch_dye_combo_var.set("All channels")

        self.GUVchannel = self._selection_to_index(self.ch_guv_combo_var.get())
        self.dye_channel = self._selection_to_index(self.ch_dye_combo_var.get())

    def on_change_guv_channel(self, event):
        self.GUVchannel = self._selection_to_index(self.ch_guv_combo.get())

    def on_change_dye_channel(self, event):
        self.dye_channel = self._selection_to_index(self.ch_dye_combo.get())

    #add file to list
    def add_file(self):
        filepaths = filedialog.askopenfilenames(title="Select CZI or TIF File", filetypes=[("CZI files", "*.czi"),("TIF files", ("*.tif", "*.TIF", "*.tiff", "*.TIFF"))])
        if filepaths:
            for file in filepaths:
                if file in self.files:
                    messagebox.showwarning("Warning", f"File {file} was already added.")
                else:
                    self.files.append(file)
                    # Add file into treeview
                    self.tree.insert("", "end", file, text=file.split("/")[-1], open=False)

    #Performs the analysis for uploaded files
    def analyze_files(self):
        if not self.files:
            messagebox.showerror("Error", "Please add at least one file!")
            return

        # Threshold-values
        try:
            mean_threshold = int(self.mean_intensity_entry.get())
            circle_radius_lower = int(self.circle_radius_lower_entry.get())
            circle_radius_upper = int(self.circle_radius_upper_entry.get())
            circle_param2 = int(self.circle_param2_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid threshold values!")
            return

        # Clear old sample/channel entries before adding new results
        for item in self.tree.get_children(""):
            for child in self.tree.get_children(item):
                self.tree.delete(child)

        # Call analysis
        self.update_channel_comboboxes()
        print("files", self.files)
        for file in self.files:
            print(f"Analyzing file {file} with thresholds: Mean={mean_threshold}, Lower Radius={circle_radius_lower}, Upper Radius={circle_radius_upper}")
            dye_intensity_sample, images_sample, individual_GUV_dye_intensity_sample = self.process_file(file, mean_threshold, circle_param2, circle_radius_lower, circle_radius_upper)
            dye_intensities_file = {}

            for sample, channels in dye_intensity_sample.items():
                dye_intensities_sample = {}
                sample_iid = f"sample::{file}::{sample}"
                if not self.tree.exists(sample_iid):
                    self.tree.insert(file, "end", iid=sample_iid, text=f"Sample {sample}")

                for GUVchannel, dye_channels in channels.items():
                    dye_intensities_sample[GUVchannel] = {}

                    guv_channel_iid = f"guvchannel::{file}::{sample}::{GUVchannel}"
                    if not self.tree.exists(guv_channel_iid):
                        self.tree.insert(sample_iid, "end", iid=guv_channel_iid, text=f"GUV Channel {GUVchannel + 1}")

                    for dye_channel, timepoints in dye_channels.items():
                        dye_intensities_sample[GUVchannel][dye_channel] = {
                            "average_intensity": timepoints,
                            "individual_intensities": individual_GUV_dye_intensity_sample[sample][GUVchannel][dye_channel]
                        }

                        dye_channel_iid = f"dyechannel::{file}::{sample}::{GUVchannel}::{dye_channel}"
                        if not self.tree.exists(dye_channel_iid):
                            self.tree.insert(guv_channel_iid, "end", iid=dye_channel_iid, text=f"Dye Channel {dye_channel + 1}")

                        for time, _ in enumerate(timepoints):
                            time_item = f"time::{file}::{sample}::{GUVchannel}::{dye_channel}::{time}"
                            if not self.tree.exists(time_item):
                                self.tree.insert(dye_channel_iid, "end", iid=time_item, text=f"Time {time}")

                dye_intensities_file[sample] = dye_intensities_sample

            self.dye_intensity_results[file] = dye_intensities_file
            self.image_results[file] = images_sample

        messagebox.showinfo("Done", "Analysis completed!")
        self.plot_results()


    #Process and analyze single file
    def process_file(self, filename, mean_threshold, circle_param2, circle_radius_lower, circle_radius_upper):
        print(filename)
        image_ch_guv_list = []
        image_ch_dye_list = []

        if filename.endswith(".czi"):
            czi = CziFile(filename)
            print(czi)

            dims_info = czi.get_dims_shape()
            print("All Scene Dims:", dims_info)

            # Wir nehmen Scene 0 (aber NICHT bei read_mosaic verwenden!)
            scene_dims = dims_info[0]
            print(scene_dims)
            if "T" in scene_dims:
                T = scene_dims['T'][1] - scene_dims['T'][0]
            if "C" in scene_dims:
                C = scene_dims['C'][1] - scene_dims['C'][0]
            if "S" in scene_dims:
                S = scene_dims['S'][1] - scene_dims['S'][0]
            else:
                S = 1
            if "M" in scene_dims:
                M = scene_dims['M'][1] - scene_dims['M'][0]
            else:
                M = 1

            print("Dims:", list(scene_dims.keys()))
            print("Shape:", {k: v[1] - v[0] for k, v in scene_dims.items()})

            total_guv_channels = int(self.GUV_channels.get())
            total_dye_channels = int(self.dye_channels.get())
            for guv in range(total_guv_channels):
                image_ch_guv_scene = []
                for scene in range(S):
                    image_ch_guv_time = []
                    for t in range(T):
                        if "S" in scene_dims and "M" in scene_dims and S > 1:
                            scene_boxes = czi.get_all_mosaic_scene_bounding_boxes()
                            region = bbox_to_region(scene_boxes[scene])
                            image_ch_guv_time.append(czi.read_mosaic(region=region, C=guv, T=t))
                        else:
                            image_ch_guv_time.append(czi.read_mosaic(C=guv, T=t))
                    image_ch_guv_scene.append(image_ch_guv_time)
                image_ch_guv_list.append(to_stxy(image_ch_guv_scene))

                image_ch_dye_per_guv = []
                for dye in range(total_dye_channels):
                    image_ch_dye_scene = []
                    dye_idx = total_guv_channels + dye
                    for scene in range(S):
                        image_ch_dye_time = []
                        for t in range(T):
                            if "S" in scene_dims and "M" in scene_dims and S > 1:
                                scene_boxes = czi.get_all_mosaic_scene_bounding_boxes()
                                region = bbox_to_region(scene_boxes[scene])
                                image_ch_dye_time.append(czi.read_mosaic(region=region, C=dye_idx, T=t))
                            else:
                                image_ch_dye_time.append(czi.read_mosaic(C=dye_idx, T=t))
                        image_ch_dye_scene.append(image_ch_dye_time)
                    image_ch_dye_per_guv.append(to_stxy(image_ch_dye_scene))
                image_ch_dye_list.append(image_ch_dye_per_guv)
                print("Channel evaluated:", guv)

        elif filename.endswith("tif") or filename.endswith("tiff") or filename.endswith("TIF") or filename.endswith("TIFF"):
            tif_data = tifffile.imread(filename)
            print(tif_data.shape)

            if tif_data.ndim < 3:
                raise ValueError("TIF input must include at least one channel and one timepoint dimension.")

            if tif_data.ndim == 3:
                tif_data = np.expand_dims(tif_data, axis=0)

            total_guv_channels = min(int(self.GUV_channels.get()), tif_data.shape[0])
            total_dye_channels = int(self.dye_channels.get())

            for guv in range(total_guv_channels):
                image_ch_guv_list.append(tif_data[guv])
                image_ch_dye_per_guv = []
                for dye in range(total_dye_channels):
                    dye_idx = min(total_guv_channels + dye, tif_data.shape[0] - 1)
                    image_ch_dye_per_guv.append(tif_data[dye_idx])
                image_ch_dye_list.append(image_ch_dye_per_guv)
                print("Channel evaluated TIF:", guv)

        dye_intensity_avg = {}
        images = {}
        dye_intensities = {}

        if self.GUVchannel == 0:
            selected_guv_channels = range(len(image_ch_guv_list))
        else:
            selected_guv_channels = [self.GUVchannel - 1] if 0 <= self.GUVchannel - 1 < len(image_ch_guv_list) else []

        print("processing circles")
        for GUVchannel in selected_guv_channels:
            image_ch_guv = image_ch_guv_list[GUVchannel]
            if self.dye_channel == 0:
                selected_dye_channels = range(len(image_ch_dye_list[GUVchannel]))
            else:
                selected_dye_channels = [self.dye_channel - 1] if 0 <= self.dye_channel - 1 < len(image_ch_dye_list[GUVchannel]) else []
            print("processing channel =", GUVchannel)

            for dye_channel in selected_dye_channels:
                image_ch_dye = image_ch_dye_list[GUVchannel][dye_channel]

                for s in range(image_ch_guv.shape[0]):
                    print("processing S =", s)
                    if s not in dye_intensity_avg:
                        dye_intensity_avg[s] = {}
                        images[s] = {}
                        dye_intensities[s] = {}
                    if GUVchannel not in dye_intensity_avg[s]:
                        dye_intensity_avg[s][GUVchannel] = {}
                        images[s][GUVchannel] = {}
                        dye_intensities[s][GUVchannel] = {}

                    dye_intensity_avg_sample = []
                    images_sample = {}
                    dye_intensities_sample = {}

                    for t in range(image_ch_guv.shape[1]):
                        print("processing t =", t)
                        dye_intensity_sum = 0
                        valid_circles_count = 0
                        dye_intensities_time = []
                        blur = cv2.GaussianBlur(image_ch_guv[s][t], (5, 5), 0)
                        dst = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        dst = clahe.apply(dst)

                        img8 = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        thr, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        dst_circles = dst.copy()
                        dst_circles[dst < thr] = 0

                        fig, ax = plt.subplots(2, 2, figsize=(3, 3), dpi=600)
                        ax[1][1].imshow(dst, cmap='gray')
                        for ax1 in ax:
                            for ax2 in ax1:
                                ax2.set_axis_off()
                                ax2.axis("off")
                        circles = cv2.HoughCircles(dst_circles, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                                   param1=50, param2=circle_param2, minRadius=circle_radius_lower, maxRadius=circle_radius_upper)

                        CRD_tolerance = float(self.circle_radius_distance_tolerance_entry.get())
                        CDTabs_tolerance = float(self.center_distance_tolerance_abs_entry.get())
                        CDTrel_tolerance = float(self.center_distance_tolerance_rel_entry.get())

                        valid_circles = []
                        whole_mask = np.zeros_like(dst)
                        if circles is not None:
                            circles = np.round(circles[0, :]).astype("int")

                            for i, (x, y, r) in enumerate(circles):
                                circle_mask = np.zeros_like(dst)
                                cv2.circle(circle_mask, (x, y), int(r * CRD_tolerance) - 3, 255, thickness=cv2.FILLED)

                                overlaps_with_any = False

                                for j, (other_x, other_y, other_r) in enumerate(circles):
                                    if i == j:
                                        continue

                                    center_distance = np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
                                    radius_diff = r - other_r
                                    if center_distance < CDTabs_tolerance + CDTrel_tolerance * r and np.abs(radius_diff < CDTabs_tolerance + CDTrel_tolerance * r) and radius_diff < 0:
                                        continue
                                    elif center_distance < (r + other_r) * CRD_tolerance:
                                        overlaps_with_any = True
                                        break

                                if not overlaps_with_any:
                                    whole_mask = cv2.bitwise_or(whole_mask, circle_mask)

                                    mean_intensity = cv2.mean(dst, mask=circle_mask)[0]
                                    ax[1][1].add_patch(plt.Circle((x, y), r + 10, color='green', fill=False, linewidth=0.2))
                                    if mean_intensity < mean_threshold:
                                        valid_circles.append((x, y, r))
                                        ax[1][1].add_patch(plt.Circle((x, y), r - 3, color='red', fill=False, linewidth=0.2))

                                        intensity_buffer = round(cv2.mean(image_ch_dye[s][t], mask=circle_mask)[0] * 100 / cv2.mean(image_ch_dye[s][t])[0], 2)
                                        if intensity_buffer > 100:
                                            intensity_buffer = 100
                                        dye_intensity_sum += intensity_buffer
                                        dye_intensities_time.append(intensity_buffer)
                                        valid_circles_count += 1

                                ax[1][0].add_patch(plt.Circle((x, y), r, color='green', fill=False, linewidth=0.4))

                            if valid_circles_count > 0:
                                normalized_intensity = dye_intensity_sum / valid_circles_count
                                dye_intensity_avg_sample.append(normalized_intensity)
                            else:
                                dye_intensity_avg_sample.append(0)
                        else:
                            dye_intensity_avg_sample.append(0)

                        ax[0][0].imshow(dst_circles > 0, cmap="gray")
                        ax[1][0].imshow(dst, cmap='gray')
                        ax[0][1].imshow(whole_mask, cmap='gray')

                        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, top=1, bottom=0)
                        fig.canvas.draw()
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        images_sample[t] = img
                        dye_intensities_sample[t] = dye_intensities_time

                        plt.close(fig)

                    dye_intensity_avg[s][GUVchannel][dye_channel] = dye_intensity_avg_sample
                    images[s][GUVchannel][dye_channel] = images_sample
                    dye_intensities[s][GUVchannel][dye_channel] = dye_intensities_sample

        return dye_intensity_avg, images, dye_intensities

    # SHow results in a plot
    def plot_results(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        for spine in ax.spines.values():
            spine.set_visible(True)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        fig.tight_layout()

        ax.set_xlabel('Time (min)', fontsize=14, labelpad=10)
        ax.set_ylabel('GUV Permeabilization (%)', fontsize=14, labelpad=10)

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1.5, length=4)

        ax.spines['top'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)

        # Time between images
        time_points = np.arange(0, 10) * 6.7 #min
        
        #export dict as json
        exportpath = os.path.dirname(self.files[0])
        export_name = f"{self.ch_guv_combo.get().replace(' ', '_')}_{self.ch_dye_combo.get().replace(' ', '_')}"
        with open(exportpath + "/data_" + export_name + ".json", "w") as json_file:
            json.dump(self.dye_intensity_results, json_file, indent=4)

        # Plot for every file
        print(self.dye_intensity_results)
        for name, file in self.dye_intensity_results.items():
            print(file)
            for sample_idx, sample in file.items():
                for channel_idx, dye_channels in sample.items():
                    for dye_channel_idx, intensity in dye_channels.items():
                        print(intensity)
                        label = f"{name.split('/')[-1]} S{sample_idx} G{int(channel_idx) + 1} D{int(dye_channel_idx) + 1}"
                        ax.plot(time_points[:len(intensity["average_intensity"])], intensity["average_intensity"], label=label, marker="o")

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.8), title="File:", frameon=False)
        fig.tight_layout()
        plt.show()

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = CZIAnalyzerApp(root)
    root.mainloop()

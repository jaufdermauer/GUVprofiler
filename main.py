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

np.set_printoptions(threshold=np.inf, linewidth=200)

# Class to create the GUI
class CZIAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUV profiler for CZI and TIF files")
        self.files = []
        self.green_intensity_results = {}
        self.image_results = {}
        self._last_width = None
        self._last_height = None

        self.root.minsize(800, 600)
        self.root.bind("<Configure>", self.on_resize)
        self.channel = 1

        self.create_widgets()
        
    def extract_time(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None  # falls keine Zahl gefunden wird

    # Resizes the image when the window size changes, with debouncing.
    def on_resize(self, event):
        selected_items = self.tree.selection()
        if selected_items and hasattr(self.image_label, "image") and self.image_label.image is not None:
            selected_item = selected_items[0]
            parent = self.tree.parent(selected_item)

            if parent:  # A specific time point is selected
                file = parent
                time_point = selected_item.split("_")[-1]  # Hier "time" in "time_point" geändert

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

                self.show_image(file, time_point)

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

        # Button to add files
        i_row = 0
        self.add_file_button = tk.Button(middle_frame, text="Add file", command=self.add_file)
        self.add_file_button.grid(row=i_row, column=0, padx=10, pady=10, sticky="ew")
        i_row+=1
        # Threshold inputs for analysis
        self.threshold_label = tk.Label(middle_frame, text="Threshold input:")
        self.threshold_label.grid(row=i_row, column=0, padx=10, pady=10, sticky="ew")
        i_row+=1

        self.mean_intensity_label = tk.Label(middle_frame, text="Maximum lipid dye intensity inside GUV:")
        self.mean_intensity_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1

        self.mean_intensity_entry = tk.Entry(middle_frame)
        self.mean_intensity_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.mean_intensity_entry.insert(0, "100")
        i_row+=1

        self.circle_radius_lower_label = tk.Label(middle_frame, text="Circle Radius lower Threshold:")
        self.circle_radius_lower_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1

        self.circle_radius_lower_entry = tk.Entry(middle_frame)
        self.circle_radius_lower_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.circle_radius_lower_entry.insert(0, "5")
        i_row+=1

        self.circle_radius_upper_label = tk.Label(middle_frame, text="Circle Radius upper Threshold:")
        self.circle_radius_upper_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1

        self.circle_radius_upper_entry = tk.Entry(middle_frame)
        self.circle_radius_upper_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.circle_radius_upper_entry.insert(0, "50")
        i_row+=1

        self.circle_param2_label = tk.Label(middle_frame, text="Detection sensitivity:")
        self.circle_param2_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1
        self.circle_param2_entry = tk.Entry(middle_frame)
        self.circle_param2_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.circle_param2_entry.insert(0, "50")
        i_row+=1

        self.circle_radius_distance_tolerance_label = tk.Label(middle_frame, text="Circle radius tolerance:")
        self.circle_radius_distance_tolerance_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1
        self.circle_radius_distance_tolerance_entry = tk.Entry(middle_frame)
        self.circle_radius_distance_tolerance_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.circle_radius_distance_tolerance_entry.insert(0, "0.9")
        i_row+=1

        self.center_distance_tolerance_abs_label = tk.Label(middle_frame, text="Circle distance tolerance (absolute):")
        self.center_distance_tolerance_abs_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1
        self.center_distance_tolerance_abs_entry = tk.Entry(middle_frame)
        self.center_distance_tolerance_abs_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.center_distance_tolerance_abs_entry.insert(0, "15")
        i_row+=1

        self.center_distance_tolerance_rel_label = tk.Label(middle_frame, text="Circle distance tolerance (relative to GUV size):")
        self.center_distance_tolerance_rel_label.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        i_row+=1
        self.center_distance_tolerance_rel_entry = tk.Entry(middle_frame)
        self.center_distance_tolerance_rel_entry.grid(row=i_row, column=0, padx=10, pady=5, sticky="ew")
        self.center_distance_tolerance_rel_entry.insert(0, "0.1")
        i_row+=1

        options = ["Channel 1", "Channel 2"]
        self.combo = ttk.Combobox(root, values=options, state="readonly")  # optional: state="readonly"
        self.combo_var = tk.StringVar(value="Channel 1")
        self.combo = ttk.Combobox(root, textvariable=self.combo_var, values=options, state="readonly")
        self.combo.bind("<<ComboboxSelected>>", self.onChangeChannel)
        self.combo.grid(row=i_row, column=0, padx=10, pady=10)
        i_row+=1

        # Button for analysis
        self.analyze_button = tk.Button(middle_frame, text="Analyze", command=self.analyze_files)
        self.analyze_button.grid(row=i_row, column=0, padx=10, pady=20, sticky="ew")
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
        selected_item = self.tree.selection()[0]
        parent = self.tree.parent(selected_item)

        if parent:  # A specific time point is selected
            file = parent
            time = selected_item.split("_")[-1]
            intensity = self.green_intensity_results[file]["average_intensity"][int(time)]
            self.show_image(file, time)
        else:
            print(f"File {selected_item} selected")

    # Displays the saved image for the selected file and time
    def show_image(self, file, time):
        selected_time_point = int(time)
        if file in self.image_results and selected_time_point in self.image_results[file]:
            image = self.image_results[file][selected_time_point]
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

    
    def onChangeChannel(self, event):
        if self.combo.get() == "Channel 1":
            self.channel = 1
        elif self.combo.get() == "Channel 2":
            self.channel = 2

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

        # Call analysis
        print("files", self.files)
        for file in self.files:
            print(f"Analyzing file {file} with thresholds: Mean={mean_threshold}, Lower Radius={circle_radius_lower}, Upper Radius={circle_radius_upper}")
            green_intensity, images, individual_GUV_green_intensity = self.process_file(file, mean_threshold, circle_param2, circle_radius_lower, circle_radius_upper)
            green_intensities_file = {}
            green_intensities_file["average_intensity"] = green_intensity
            green_intensities_file["individual_intensities"] = individual_GUV_green_intensity
            self.green_intensity_results[file] = green_intensities_file
            self.image_results[file] = images  # Save images for files here
            for time, intensity in enumerate(green_intensity):
                time_item = f"{file}_{time}"
                # Prevent double entries
                if not self.tree.exists(time_item):
                    self.tree.insert(file, "end", f"{file}_{time}", text=f"Time {time}")
        messagebox.showinfo("Done", "Analysis completed!")
        self.plot_results()


    #Process and analyze single file
    def process_file(self, filename, mean_threshold, circle_param2, circle_radius_lower, circle_radius_upper):
        print(filename)
        if filename.endswith(".czi"):
            czi = CziFile(filename)

            dims_info = czi.get_dims_shape()
            print("All Scene Dims:", dims_info)

            # Wir nehmen Scene 0 (aber NICHT bei read_mosaic verwenden!)
            scene_dims = dims_info[0]
            T = scene_dims['T'][1] - scene_dims['T'][0]
            C = scene_dims['C'][1] - scene_dims['C'][0]

            print("Dims:", list(scene_dims.keys()))
            print("Shape:", {k: v[1] - v[0] for k, v in scene_dims.items()})

            image_ch0_list = []
            image_ch1_list = []

            for t in range(T):
                image_ch0_list.append(czi.read_mosaic(C=0, T=t))
                try:
                    image_ch1_list.append(czi.read_mosaic(C=self.channel, T=t))
                    print("Channel evaluated:", channel)
                except:
                    print("Kein gültiger Channel")

            image_ch0 = np.stack(image_ch0_list, axis=0)[0, 0, :, :]
            image_ch1 = np.stack(image_ch1_list, axis=0)[0, 0, :, :]

                
        elif filename.endswith("tif") or filename.endswith("tiff") or filename.endswith("TIF") or filename.endswith("TIFF"):
            image_ch0_list = []
            image_ch1_list = []

            print(tifffile.imread(filename).shape)

            
            image_ch0_list.append(tifffile.imread(filename)[0])
            try:
                image_ch1_list.append(tifffile.imread(filename)[self.channel])
                print("Channel evaluated TIF:", channel)
            except:
                print("Kein gültiger Channel TIF")

            image_ch0 = np.stack(image_ch0_list, axis=0)
            image_ch1 = np.stack(image_ch1_list, axis=0)
        green_intensity_avg = []
        images = {}
        green_intensities = {}

        for t in range(image_ch0.shape[0]):
            # Variables for green intensity signal
            green_intensity_sum = 0
            valid_circles_count = 0
            green_intensities_time = []
            blur = cv2.GaussianBlur(image_ch0[t], (5,5), 0)
            dst = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            dst = clahe.apply(dst)
            dst_circles = dst
            dst_circles[dst_circles<40] = 0
            print(dst.shape)
            fig, ax = plt.subplots(2, 2, figsize=(3,3), dpi=600)
            ax[1][1].imshow(dst, cmap='gray')
            for ax1 in ax:
                for ax2 in ax1:
                    ax2.set_axis_off()
                    ax2.axis("off")
            circles = cv2.HoughCircles(dst_circles, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                    param1=50, param2=circle_param2, minRadius=circle_radius_lower, maxRadius=circle_radius_upper)

            # If there are circles
            CRD_tolerance = float(self.circle_radius_distance_tolerance_entry.get())
            CDTabs_tolerance = float(self.center_distance_tolerance_abs_entry.get())
            CDTrel_tolerance = float(self.center_distance_tolerance_rel_entry.get())

            valid_circles = []
            whole_mask = np.zeros_like(dst)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

        
                # Check and plot circles
                for i, (x, y, r) in enumerate(circles):
                    # Mask for circle
                    circle_mask = np.zeros_like(dst)
                    cv2.circle(circle_mask, (x, y), int(r*CRD_tolerance)-3, 255, thickness=cv2.FILLED)

                    overlaps_with_any = False

                    # Check for overlap between circles
                    for j, (other_x, other_y, other_r) in enumerate(circles):
                        if i == j:
                            continue

                        center_distance = np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
                        radius_diff = r - other_r
                        if center_distance < CDTabs_tolerance + CDTrel_tolerance*r and np.abs(radius_diff < CDTabs_tolerance + CDTrel_tolerance*r) and radius_diff < 0:
                            print("ring detected")
                        elif center_distance < (r + other_r)*CRD_tolerance:
                            overlaps_with_any = True
                            break

                    if not overlaps_with_any:
                        whole_mask = cv2.bitwise_or(whole_mask, circle_mask)

                        # Calculate intensity within circles
                        mean_intensity = cv2.mean(dst, mask=circle_mask)[0]
                        ax[1][1].add_patch(plt.Circle((x, y), r + 10, color='green', fill=False, linewidth=0.2))
                        if mean_intensity < mean_threshold:
                            valid_circles.append((x, y, r))
                            ax[1][1].add_patch(plt.Circle((x, y), r - 3, color='red', fill=False, linewidth=0.2))

                            # Add green intensity to buffer dict
                            intensity_buffer = round(cv2.mean(image_ch1[t], mask=circle_mask)[0] * 100 / cv2.mean(image_ch1[t])[0],2)
                            if intensity_buffer > 100:
                                intensity_buffer = 100
                            green_intensity_sum += intensity_buffer
                            green_intensities_time.append(intensity_buffer)
                            valid_circles_count += 1

                    ax[1][0].add_patch(plt.Circle((x, y), r, color='green', fill=False, linewidth=0.4))

                # Normalize and add to green_intensity_avg
                if valid_circles_count > 0:
                    normalized_intensity = green_intensity_sum / valid_circles_count
                    green_intensity_avg.append(normalized_intensity)
                else:
                    green_intensity_avg.append(0)  # If no circles are found
            else:
                green_intensity_avg.append(0)  # If no circles are found
            
            print("show images")
            ax[0][0].imshow(dst, cmap='gray')
            ax[1][0].imshow(dst, cmap='gray')
            ax[0][1].imshow(whole_mask, cmap='gray')

            # Save figure in dictionary
            plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images[t] = img
            green_intensities[t] = green_intensities_time

            plt.close(fig)  # Close figure to save memory

        return green_intensity_avg, images, green_intensities

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
        with open(exportpath + "/data.json", "w") as json_file:
            json.dump(self.green_intensity_results, json_file, indent=4)

        # Plot for every file
        for file, intensity in self.green_intensity_results.items():
            label = file.split("/")[-1]
            ax.plot(time_points[:len(intensity["average_intensity"])], intensity["average_intensity"], label=label, marker="o")

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.8), title="Peptide", frameon=False)
        fig.tight_layout()
        plt.show()

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = CZIAnalyzerApp(root)
    root.mainloop()

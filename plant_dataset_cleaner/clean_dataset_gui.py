# clean_dataset_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from clean_dataset_cli import clean_dataset


class DatasetCleanerGUI:
    def __init__(self, root):
        self.root = root
        root.title("植物病害数据集清洗工具")
        root.geometry("600x200")

        tk.Label(root, text="选择你的数据集根目录:").pack(pady=10)

        self.path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.path_var, width=70).pack(pady=5)
        tk.Button(root, text="浏览...", command=self.browse_folder).pack(pady=5)

        self.dry_run_var = tk.BooleanVar()
        tk.Checkbutton(root, text="仅预览（不修改文件）", variable=self.dry_run_var).pack(pady=5)

        self.run_btn = tk.Button(root, text="开始清洗", command=self.start_cleaning)
        self.run_btn.pack(pady=10)

        self.progress = ttk.Progressbar(root, mode='indeterminate')

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)

    def start_cleaning(self):
        path = self.path_var.get()
        if not path:
            messagebox.showerror("错误", "请选择数据集目录！")
            return

        self.run_btn.config(state="disabled")
        self.progress.pack(pady=10)
        self.progress.start()

        def task():
            try:
                clean_dataset(path, dry_run=self.dry_run_var.get())
                self.root.after(0, lambda: messagebox.showinfo("完成", "数据集清洗完成！"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
            finally:
                self.root.after(0, self.cleanup)

        threading.Thread(target=task, daemon=True).start()

    def cleanup(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.run_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCleanerGUI(root)
    root.mainloop()
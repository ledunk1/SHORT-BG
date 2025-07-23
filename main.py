#!/usr/bin/env python3
"""
Green Screen Auto Editor - Main Application
Entry point for the application
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
"""
Flet-based GUI for HRAudioWizard.

Modern, cross-platform UI with file list management and i18n support.
"""

import os
import flet as ft
import threading
import traceback

# Import with fallback
try:
    from .localization import Localization
    from .hfr import HighFrequencyRestorer
    from .hfp_v1 import HighFrequencyRestorerV1
    from .noise_shaper import RDONoiseShaper
    from .resampler import IntelligentRDOResampler
except ImportError:
    from localization import Localization
    from hfr import HighFrequencyRestorer
    from hfp_v1 import HighFrequencyRestorerV1
    from noise_shaper import RDONoiseShaper
    from resampler import IntelligentRDOResampler

import numpy as np
import librosa
import soundfile as sf


class AudioWizardApp:
    """Main Flet application class."""
    
    def __init__(self):
        self.loc = Localization("ja")
        self.file_list = []
        self.output_folder = ""
        self.is_processing = False
        self.stop_requested = False
        
    def build(self, page: ft.Page):
        """Build the main UI."""
        self.page = page
        
        # Page settings
        page.title = self.loc.get("app_title")
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 20
        page.window.width = 900
        page.window.height = 800
        page.bgcolor = "#1a1a2e"
        
        # Custom theme
        page.theme = ft.Theme(
            color_scheme=ft.ColorScheme(
                primary="#6c5ce7",
                secondary="#a29bfe",
                surface="#16213e",
                background="#1a1a2e",
            )
        )
        
        # Build UI components
        self._build_header()
        self._build_file_list()
        self._build_output_section()
        self._build_settings()
        self._build_action_section()
        self._build_log()
        
        # Main layout
        page.add(
            ft.Container(
                content=ft.Column([
                    self.header,
                    ft.Divider(color="#333"),
                    self.file_section,
                    self.output_section,
                    ft.Divider(color="#333"),
                    self.settings_section,
                    ft.Divider(color="#333"),
                    self.action_section,
                    self.log_section,
                ], spacing=15, scroll=ft.ScrollMode.AUTO),
                expand=True,
            )
        )
    
    def _build_header(self):
        """Build header with title and language selector."""
        self.lang_dropdown = ft.Dropdown(
            value="ja",
            options=[
                ft.dropdown.Option("ja", "日本語"),
                ft.dropdown.Option("en", "English"),
            ],
            width=120,
            on_change=self._on_language_change,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        self.header = ft.Row([
            ft.Text(
                self.loc.get("app_title"),
                size=28,
                weight=ft.FontWeight.BOLD,
                color="#6c5ce7",
            ),
            ft.Container(expand=True),
            ft.Text(self.loc.get("language"), color="#888"),
            self.lang_dropdown,
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    
    def _build_file_list(self):
        """Build file list section with add/delete buttons."""
        self.file_list_view = ft.ListView(
            expand=True,
            spacing=5,
            height=200,
            auto_scroll=True,
        )
        
        self.file_list_container = ft.Container(
            content=self.file_list_view,
            bgcolor="#16213e",
            border_radius=10,
            padding=10,
            border=ft.border.all(1, "#333"),
        )
        
        # Empty state
        self.empty_text = ft.Text(
            self.loc.get("no_files"),
            color="#666",
            italic=True,
            text_align=ft.TextAlign.CENTER,
        )
        self.file_list_view.controls.append(
            ft.Container(content=self.empty_text, alignment=ft.alignment.center, height=180)
        )
        
        # Buttons
        self.add_file_btn = ft.ElevatedButton(
            self.loc.get("add_file"),
            icon=ft.Icons.ADD,
            on_click=self._on_add_file,
            bgcolor="#6c5ce7",
            color="white",
        )
        
        self.add_folder_btn = ft.ElevatedButton(
            self.loc.get("add_folder"),
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_add_folder,
            bgcolor="#5a52d5",
            color="white",
        )
        
        self.delete_btn = ft.ElevatedButton(
            self.loc.get("delete"),
            icon=ft.Icons.DELETE,
            on_click=self._on_delete_selected,
            bgcolor="#e74c3c",
            color="white",
        )
        
        self.delete_all_btn = ft.ElevatedButton(
            self.loc.get("delete_all"),
            icon=ft.Icons.DELETE_FOREVER,
            on_click=self._on_delete_all,
            bgcolor="#c0392b",
            color="white",
        )
        
        button_row = ft.Row([
            self.add_file_btn,
            self.add_folder_btn,
            ft.Container(expand=True),
            self.delete_btn,
            self.delete_all_btn,
        ], spacing=10)
        
        self.file_section = ft.Container(
            content=ft.Column([
                ft.Text(self.loc.get("files_to_process"), size=16, weight=ft.FontWeight.BOLD, color="#a29bfe"),
                self.file_list_container,
                button_row,
            ], spacing=10),
        )
        
        # File picker
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        self.folder_picker = ft.FilePicker(on_result=self._on_folder_picked)
        self.output_picker = ft.FilePicker(on_result=self._on_output_picked)
        self.page.overlay.extend([self.file_picker, self.folder_picker, self.output_picker])
    
    def _build_output_section(self):
        """Build output folder selection."""
        self.output_text = ft.TextField(
            label=self.loc.get("output_folder"),
            read_only=True,
            expand=True,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        self.output_btn = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_select_output,
            icon_color="#6c5ce7",
        )
        
        self.output_section = ft.Row([
            self.output_text,
            self.output_btn,
        ], spacing=10)
    
    def _build_settings(self):
        """Build settings panel."""
        # Algorithm
        self.algo_dropdown = ft.Dropdown(
            label=self.loc.get("algorithm"),
            value="HFPv2 (MDCT-based)",
            options=[
                ft.dropdown.Option("HFPv2 (MDCT-based)"),
                ft.dropdown.Option("HFP V1 (STFT-based)"),
            ],
            width=200,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        # Sample Rate
        self.sr_dropdown = ft.Dropdown(
            label=self.loc.get("sample_rate"),
            value="48000",
            options=[ft.dropdown.Option(sr) for sr in ["32000", "44100", "48000", "88200", "96000", "192000"]],
            width=150,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        # Bit Depth
        self.bit_dropdown = ft.Dropdown(
            label=self.loc.get("bit_depth"),
            value="24-bit PCM",
            options=[ft.dropdown.Option(bd) for bd in ["16-bit PCM", "24-bit PCM", "32-bit Float"]],
            width=150,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        # Checkboxes
        self.dither_check = ft.Checkbox(
            label=self.loc.get("enable_dither"),
            value=True,
            active_color="#6c5ce7",
        )
        
        self.hfr_check = ft.Checkbox(
            label=self.loc.get("enable_hfr"),
            value=True,
            active_color="#6c5ce7",
        )
        
        self.compressed_check = ft.Checkbox(
            label=self.loc.get("compressed_fix"),
            value=True,
            active_color="#6c5ce7",
        )
        
        # Lowpass
        self.lowpass_field = ft.TextField(
            label=self.loc.get("lowpass_cutoff"),
            value="13500",
            width=120,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        # LPC Order
        self.lpc_field = ft.TextField(
            label=self.loc.get("lpc_order"),
            value="16",
            width=80,
            border_color="#6c5ce7",
            bgcolor="#16213e",
        )
        
        self.settings_section = ft.Container(
            content=ft.Column([
                ft.Text(self.loc.get("settings"), size=16, weight=ft.FontWeight.BOLD, color="#a29bfe"),
                ft.Row([self.algo_dropdown, self.sr_dropdown, self.bit_dropdown], spacing=15),
                ft.Row([self.dither_check, self.hfr_check, self.compressed_check], spacing=15),
                ft.Row([self.lowpass_field, self.lpc_field], spacing=15),
            ], spacing=10),
        )
    
    def _build_action_section(self):
        """Build start/stop button and progress."""
        self.start_btn = ft.ElevatedButton(
            self.loc.get("start_processing"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start_stop,
            bgcolor="#00b894",
            color="white",
            width=250,
            height=50,
        )
        
        self.progress_bar = ft.ProgressBar(
            value=0,
            width=400,
            color="#6c5ce7",
            bgcolor="#333",
        )
        
        self.progress_text = ft.Text(
            self.loc.get("ready"),
            color="#888",
        )
        
        self.action_section = ft.Column([
            ft.Row([self.start_btn], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.progress_bar], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.progress_text], alignment=ft.MainAxisAlignment.CENTER),
        ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    
    def _build_log(self):
        """Build log console."""
        self.log_text = ft.TextField(
            multiline=True,
            read_only=True,
            min_lines=6,
            max_lines=10,
            border_color="#333",
            bgcolor="#0f0f1a",
            text_style=ft.TextStyle(font_family="Consolas", size=12),
        )
        
        self.log_section = ft.Container(
            content=ft.Column([
                ft.Text(self.loc.get("log"), size=14, color="#888"),
                self.log_text,
            ], spacing=5),
        )
    
    # === Event Handlers ===
    
    def _on_language_change(self, e):
        """Handle language change."""
        self.loc.set_language(e.control.value)
        self._update_all_labels()
        self.page.update()
    
    def _update_all_labels(self):
        """Update all UI labels for current language."""
        # Update texts
        self.page.title = self.loc.get("app_title")
        self.add_file_btn.text = self.loc.get("add_file")
        self.add_folder_btn.text = self.loc.get("add_folder")
        self.delete_btn.text = self.loc.get("delete")
        self.delete_all_btn.text = self.loc.get("delete_all")
        self.output_text.label = self.loc.get("output_folder")
        self.algo_dropdown.label = self.loc.get("algorithm")
        self.sr_dropdown.label = self.loc.get("sample_rate")
        self.bit_dropdown.label = self.loc.get("bit_depth")
        self.dither_check.label = self.loc.get("enable_dither")
        self.hfr_check.label = self.loc.get("enable_hfr")
        self.compressed_check.label = self.loc.get("compressed_fix")
        self.lowpass_field.label = self.loc.get("lowpass_cutoff")
        self.lpc_field.label = self.loc.get("lpc_order")
        self.start_btn.text = self.loc.get("start_processing") if not self.is_processing else self.loc.get("stop_processing")
    
    def _on_add_file(self, e):
        """Handle add file button."""
        self.file_picker.pick_files(
            dialog_title=self.loc.get("file_dialog_title"),
            allowed_extensions=["wav", "flac", "mp3", "ogg", "aiff"],
            allow_multiple=True,
        )
    
    def _on_add_folder(self, e):
        """Handle add folder button."""
        self.folder_picker.get_directory_path(
            dialog_title=self.loc.get("folder_dialog_title"),
        )
    
    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        """Handle file picker result."""
        if e.files:
            for f in e.files:
                if f.path not in self.file_list:
                    self.file_list.append(f.path)
            self._refresh_file_list()
    
    def _on_folder_picked(self, e: ft.FilePickerResultEvent):
        """Handle folder picker result."""
        if e.path:
            supported = ('.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif')
            for root, _, files in os.walk(e.path):
                for file in files:
                    if file.lower().endswith(supported):
                        path = os.path.join(root, file)
                        if path not in self.file_list:
                            self.file_list.append(path)
            self._refresh_file_list()
    
    def _on_select_output(self, e):
        """Handle output folder selection."""
        self.output_picker.get_directory_path(
            dialog_title=self.loc.get("select_output"),
        )
    
    def _on_output_picked(self, e: ft.FilePickerResultEvent):
        """Handle output folder picker result."""
        if e.path:
            self.output_folder = e.path
            self.output_text.value = e.path
            self.page.update()
    
    def _on_delete_selected(self, e):
        """Delete selected files."""
        # For simplicity, delete last added file
        if self.file_list:
            self.file_list.pop()
            self._refresh_file_list()
    
    def _on_delete_all(self, e):
        """Delete all files."""
        self.file_list.clear()
        self._refresh_file_list()
        self._log(self.loc.get("files_cleared"))
    
    def _refresh_file_list(self):
        """Refresh the file list view."""
        self.file_list_view.controls.clear()
        
        if not self.file_list:
            self.file_list_view.controls.append(
                ft.Container(content=self.empty_text, alignment=ft.alignment.center, height=180)
            )
        else:
            for i, path in enumerate(self.file_list):
                self.file_list_view.controls.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.AUDIO_FILE, color="#6c5ce7"),
                            ft.Text(os.path.basename(path), expand=True, color="white"),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                icon_color="#e74c3c",
                                data=i,
                                on_click=self._on_delete_item,
                            ),
                        ]),
                        bgcolor="#1e1e3f",
                        border_radius=5,
                        padding=5,
                    )
                )
        
        self.page.update()
    
    def _on_delete_item(self, e):
        """Delete specific item."""
        idx = e.control.data
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            self._refresh_file_list()
    
    def _on_start_stop(self, e):
        """Handle start/stop button."""
        if self.is_processing:
            self.stop_requested = True
            self.start_btn.text = self.loc.get("start_processing")
            self.start_btn.bgcolor = "#00b894"
        else:
            if not self.file_list:
                self._log(self.loc.get("no_files_selected"))
                return
            if not self.output_folder:
                self._log(self.loc.get("no_output_folder"))
                return
            
            self.is_processing = True
            self.stop_requested = False
            self.start_btn.text = self.loc.get("stop_processing")
            self.start_btn.bgcolor = "#e74c3c"
            self.page.update()
            
            # Start processing thread
            threading.Thread(target=self._process_files, daemon=True).start()
    
    def _process_files(self):
        """Process files in background thread."""
        try:
            total = len(self.file_list)
            
            for i, file_path in enumerate(self.file_list):
                if self.stop_requested:
                    self._log(self.loc.get("cancelled"))
                    break
                
                self._log(f"--- [{i+1}/{total}] {os.path.basename(file_path)} ---")
                self._update_progress((i / total) * 100, f"Processing {i+1}/{total}")
                
                # Load audio
                self._log(self.loc.get("loading_file"))
                dat, sr = librosa.load(file_path, sr=None, mono=False)
                dat = dat.T
                
                if dat.ndim != 2 or dat.shape[1] != 2:
                    self._log(f"[Skip] Not stereo")
                    continue
                
                processed = dat.astype(np.float64)
                target_sr = int(self.sr_dropdown.value)
                
                # Resample
                if sr != target_sr:
                    self._log(self.loc.get("resampling"))
                    resampler = IntelligentRDOResampler(sr, target_sr)
                    mid = (processed[:, 0] + processed[:, 1]) / 2
                    side = (processed[:, 0] - processed[:, 1]) / 2
                    resampled = resampler.resample(np.stack([mid, side], axis=1), True)
                    processed = np.stack([
                        resampled[:, 0] + resampled[:, 1],
                        resampled[:, 0] - resampled[:, 1]
                    ], axis=1)
                    sr = target_sr
                
                # HFR
                if self.hfr_check.value:
                    self._log(self.loc.get("hfr_processing"))
                    lowpass = int(self.lowpass_field.value) if self.lowpass_field.value != "" else -1
                    
                    if "V1" in self.algo_dropdown.value:
                        restorer = HighFrequencyRestorerV1()
                        # Set progress callback for V1
                        restorer.progress_callback = self._update_progress
                        processed = restorer.run_hfp_v1(processed, sr, lowpass, self.compressed_check.value)
                    else:
                        restorer = HighFrequencyRestorer()
                        # Connect progress signal for HFPv2
                        restorer.progress_updated.connect(self._update_progress)
                        lowpass_bin = int(lowpass * 1024 / sr) if lowpass != -1 else -1
                        processed = restorer.run_hfpv2(processed, sr, lowpass_bin, self.compressed_check.value)
                
                # Dither
                if self.dither_check.value and "PCM" in self.bit_dropdown.value:
                    self._log(self.loc.get("dithering"))
                    bit_depth = int(self.bit_dropdown.value.split("-")[0])
                    lpc = int(self.lpc_field.value)
                    shaper = RDONoiseShaper(bit_depth, sr, lpc_order=lpc)
                    processed = shaper.process(processed)
                
                # Save
                self._log(self.loc.get("saving"))
                out_path = os.path.join(self.output_folder, os.path.basename(file_path))
                subtype = "FLOAT" if "Float" in self.bit_dropdown.value else f"PCM_{self.bit_dropdown.value.split('-')[0]}"
                sf.write(out_path + ".wav", np.clip(processed, -1, 1), sr, subtype=subtype)
            
            self._log(self.loc.get("processing_complete"))
            self._update_progress(100, self.loc.get("complete"))
            
        except Exception as ex:
            self._log(f"{self.loc.get('error')}: {ex}")
            traceback.print_exc()
        
        finally:
            self.is_processing = False
            self._safe_update(self._reset_ui)
    
    def _reset_ui(self):
        """Reset UI after processing."""
        self.start_btn.text = self.loc.get("start_processing")
        self.start_btn.bgcolor = "#00b894"
        self.page.update()
    
    def _update_progress(self, value, text):
        """Update progress bar from thread."""
        self.progress_bar.value = value / 100
        self.progress_text.value = text
        self._safe_update()
    
    def _log(self, msg):
        """Add log message."""
        current = self.log_text.value or ""
        self.log_text.value = current + msg + "\n"
        self._safe_update()
    
    def _safe_update(self, callback=None):
        """Thread-safe page update."""
        try:
            if callback:
                callback()
            self.page.update()
        except Exception:
            pass  # Ignore update errors from background thread


def main(page: ft.Page):
    """Flet app entry point."""
    app = AudioWizardApp()
    app.build(page)


if __name__ == "__main__":
    ft.app(target=main)

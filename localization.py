"""
Localization module for multi-language support.

Supports Japanese (日本語) and English.
"""

# Language strings dictionary
STRINGS = {
    "ja": {
        # App
        "app_title": "🎵 HRAudioWizard",
        "language": "言語",
        
        # File list
        "files_to_process": "変換するファイル",
        "add_file": "ファイル追加",
        "add_folder": "フォルダ追加",
        "delete": "削除",
        "delete_all": "全て削除",
        "no_files": "ファイルがありません",
        "file_dialog_title": "音声ファイルを選択",
        "folder_dialog_title": "フォルダを選択",
        
        # Output
        "output_folder": "出力フォルダ",
        "select_output": "出力先を選択",
        
        # Settings
        "settings": "⚙️ 設定",
        "algorithm": "アルゴリズム",
        "sample_rate": "サンプルレート",
        "bit_depth": "ビット深度",
        "enable_dither": "ディザリング有効",
        "enable_hfr": "高周波復元有効",
        "lowpass_cutoff": "ローパス周波数",
        "auto": "自動",
        "compressed_fix": "圧縮音源修復",
        "lpc_order": "LPC次数",
        
        # Actions
        "start_processing": "▶ 処理開始",
        "stop_processing": "⬛ 停止",
        "processing": "処理中...",
        
        # Log
        "log": "ログ",
        "ready": "準備完了",
        "loading_file": "ファイル読み込み中...",
        "resampling": "リサンプリング中...",
        "hfr_processing": "高周波復元中...",
        "dithering": "ディザリング中...",
        "saving": "保存中...",
        "complete": "完了!",
        "error": "エラー",
        "cancelled": "キャンセルされました",
        
        # Messages
        "no_files_selected": "ファイルが選択されていません",
        "no_output_folder": "出力フォルダが選択されていません",
        "processing_complete": "全ての処理が完了しました！",
        "file_added": "ファイルを追加しました",
        "files_cleared": "リストをクリアしました",
    },
    
    "en": {
        # App
        "app_title": "🎵 HRAudioWizard",
        "language": "Language",
        
        # File list
        "files_to_process": "Files to Process",
        "add_file": "Add File",
        "add_folder": "Add Folder",
        "delete": "Delete",
        "delete_all": "Delete All",
        "no_files": "No files",
        "file_dialog_title": "Select Audio File",
        "folder_dialog_title": "Select Folder",
        
        # Output
        "output_folder": "Output Folder",
        "select_output": "Select Output",
        
        # Settings
        "settings": "⚙️ Settings",
        "algorithm": "Algorithm",
        "sample_rate": "Sample Rate",
        "bit_depth": "Bit Depth",
        "enable_dither": "Enable Dither",
        "enable_hfr": "Enable HFR",
        "lowpass_cutoff": "Lowpass Cutoff",
        "auto": "Auto",
        "compressed_fix": "Compressed Fix",
        "lpc_order": "LPC Order",
        
        # Actions
        "start_processing": "▶ Start Processing",
        "stop_processing": "⬛ Stop",
        "processing": "Processing...",
        
        # Log
        "log": "Log",
        "ready": "Ready",
        "loading_file": "Loading file...",
        "resampling": "Resampling...",
        "hfr_processing": "HFR Processing...",
        "dithering": "Dithering...",
        "saving": "Saving...",
        "complete": "Complete!",
        "error": "Error",
        "cancelled": "Cancelled",
        
        # Messages
        "no_files_selected": "No files selected",
        "no_output_folder": "No output folder selected",
        "processing_complete": "All processing complete!",
        "file_added": "File added",
        "files_cleared": "List cleared",
    }
}


class Localization:
    """Localization helper class."""
    
    def __init__(self, language="ja"):
        self.language = language
    
    def get(self, key):
        """Get localized string."""
        return STRINGS.get(self.language, STRINGS["en"]).get(key, key)
    
    def set_language(self, language):
        """Set current language."""
        if language in STRINGS:
            self.language = language
    
    @property
    def available_languages(self):
        """Get available language codes."""
        return list(STRINGS.keys())
    
    @property
    def language_names(self):
        """Get language display names."""
        return {"ja": "日本語", "en": "English"}
